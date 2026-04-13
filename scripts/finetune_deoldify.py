from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms.transforms import RandomResizedCrop

from src.pipeline.model_arch import DeoldifyVideoModel
from src.pipeline.model_loader import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ManifestSample:
    split: str
    movie: str
    image_path: str
    sample_index: int
    timestamp_seconds: float


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_rgb_l1: float
    train_chroma_l1: float
    train_saturation_l1: float
    train_blue_penalty: float
    val_loss: float
    val_rgb_l1: float
    val_chroma_l1: float
    val_saturation_l1: float
    val_blue_penalty: float
    epoch_seconds: float


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_val_loss: float
    history: list[EpochMetrics]
    base_checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised DeOldify generator fine-tuning with extra pressure against unwanted blue casts."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl from prepare_finetune_dataset.py.")
    parser.add_argument(
        "--checkpoint",
        default="models/deoldify/ColorizeVideo_gen.pth",
        help="Base DeOldify generator checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/deoldify/finetune",
        help="Directory for checkpoints, metrics, and previews.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of fine-tuning epochs.")
    parser.add_argument("--image-size", type=int, default=256, help="Square crop size used during training.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Decoder learning rate.")
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.25,
        help="Multiplier applied to the encoder learning rate.",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=1,
        help="Keep the ResNet encoder frozen for the first N epochs.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu"],
        help="Training device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed.",
    )
    parser.add_argument(
        "--rgb-loss-weight",
        type=float,
        default=1.0,
        help="Weight for RGB reconstruction loss.",
    )
    parser.add_argument(
        "--chroma-loss-weight",
        type=float,
        default=1.10,
        help="Weight for chroma reconstruction loss.",
    )
    parser.add_argument(
        "--saturation-loss-weight",
        type=float,
        default=0.20,
        help="Weight for saturation magnitude matching.",
    )
    parser.add_argument(
        "--blue-penalty-weight",
        type=float,
        default=0.08,
        help="Weight for penalizing excess blue where the target is not blue-dominant.",
    )
    parser.add_argument(
        "--blue-margin",
        type=float,
        default=0.02,
        help="Tolerance before the blue penalty activates.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="How many validation samples to render into each epoch preview image.",
    )
    parser.add_argument(
        "--center-loss-strength",
        type=float,
        default=2.0,
        help="Extra loss weight applied to central image regions.",
    )
    parser.add_argument(
        "--center-loss-sigma",
        type=float,
        default=0.45,
        help="Gaussian spread used for the center-prior loss weighting.",
    )
    parser.add_argument(
        "--neutral-chroma-penalty-weight",
        type=float,
        default=0.12,
        help="Penalty for adding chroma where the target is near-neutral.",
    )
    parser.add_argument(
        "--neutral-saturation-threshold",
        type=float,
        default=0.10,
        help="Target saturation threshold below which neutral-region penalties activate.",
    )
    parser.add_argument(
        "--skin-blue-penalty-weight",
        type=float,
        default=0.35,
        help="Extra blue-avoidance penalty applied to likely skin regions.",
    )
    parser.add_argument(
        "--costume-blue-penalty-weight",
        type=float,
        default=0.18,
        help="Extra blue-avoidance penalty applied to likely costume regions.",
    )
    parser.add_argument(
        "--costume-dark-penalty-weight",
        type=float,
        default=0.20,
        help="Penalty for making likely costume regions darker than the target.",
    )
    parser.add_argument(
        "--costume-dark-margin",
        type=float,
        default=0.08,
        help="Tolerance before the costume-dark penalty activates.",
    )
    parser.add_argument(
        "--costume-vividness-penalty-weight",
        type=float,
        default=0.28,
        help="Penalty for under-saturated costume regions that should carry color.",
    )
    parser.add_argument(
        "--costume-vividness-threshold",
        type=float,
        default=0.18,
        help="Target saturation threshold above which costume vividness is enforced.",
    )
    parser.add_argument(
        "--costume-vividness-margin",
        type=float,
        default=0.03,
        help="Tolerance before the costume vividness penalty activates.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/training_state.pth if it exists.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Explicit training_state checkpoint path to resume from.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save an inference-compatible checkpoint for every completed epoch.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    seed_everything(args.seed)

    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(manifest_path)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    if not train_samples:
        raise RuntimeError("No train samples found in manifest.")
    if not val_samples:
        raise RuntimeError("No validation samples found in manifest.")

    device = resolve_device(args.device)
    log(f"Device: {device}")
    log(f"Train samples: {len(train_samples)}")
    log(f"Val samples:   {len(val_samples)}")

    train_dataset = MovieColorDataset(train_samples, image_size=args.image_size, train=True)
    val_dataset = MovieColorDataset(val_samples, image_size=args.image_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = load_model(checkpoint_path, device)
    encoder_parameters = list(model[0].parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.learning_rate * args.encoder_lr_scale},
            {"params": decoder_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    metrics_log_path = output_dir / "metrics.jsonl"
    best_checkpoint_path = output_dir / "best.pth"
    last_checkpoint_path = output_dir / "last.pth"
    epoch_checkpoint_dir = output_dir / "epoch_checkpoints"
    training_state_path = output_dir / "training_state.pth"
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    if args.save_every_epoch:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_state = maybe_resume_state(
        args=args,
        checkpoint_path=checkpoint_path,
        training_state_path=training_state_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    best_val_loss = resume_state.best_val_loss
    preview_batch = next(iter(val_loader))
    history = resume_state.history
    base_checkpoint_path = resume_state.base_checkpoint_path
    if resume_state.start_epoch > args.epochs:
        log(f"Training already complete at epoch {resume_state.start_epoch - 1}. Nothing to do.")
        return 0
    if resume_state.start_epoch > 1:
        log(f"Resuming from epoch {resume_state.start_epoch}.")

    for epoch in range(resume_state.start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        set_module_trainable(model[0], epoch > args.freeze_encoder_epochs)
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            args=args,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            optimizer=None,
            args=args,
            train=False,
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_rgb_l1=train_metrics["rgb_l1"],
            train_chroma_l1=train_metrics["chroma_l1"],
            train_saturation_l1=train_metrics["saturation_l1"],
            train_blue_penalty=train_metrics["blue_penalty"],
            val_loss=val_metrics["loss"],
            val_rgb_l1=val_metrics["rgb_l1"],
            val_chroma_l1=val_metrics["chroma_l1"],
            val_saturation_l1=val_metrics["saturation_l1"],
            val_blue_penalty=val_metrics["blue_penalty"],
            epoch_seconds=epoch_seconds,
        )
        history.append(metrics)
        append_jsonl(metrics_log_path, asdict(metrics))
        render_preview(
            model=model,
            batch=preview_batch,
            device=device,
            output_path=preview_dir / f"epoch_{epoch:02d}.png",
            preview_count=args.preview_count,
        )
        save_checkpoint(last_checkpoint_path, model, args, epoch, metrics)
        if args.save_every_epoch:
            save_checkpoint(epoch_checkpoint_dir / f"epoch_{epoch:02d}.pth", model, args, epoch, metrics)
        save_training_state(
            checkpoint_path=training_state_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            epoch=epoch,
            best_val_loss=best_val_loss,
            history=history,
            base_checkpoint_path=base_checkpoint_path,
        )
        if metrics.val_loss < best_val_loss:
            best_val_loss = metrics.val_loss
            save_checkpoint(best_checkpoint_path, model, args, epoch, metrics)
            save_training_state(
                checkpoint_path=training_state_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                best_val_loss=best_val_loss,
                history=history,
                base_checkpoint_path=base_checkpoint_path,
            )
        write_summary(
            summary_path=output_dir / "summary.json",
            best_val_loss=best_val_loss,
            history=history,
            manifest_path=manifest_path,
            base_checkpoint_path=base_checkpoint_path,
        )
        log(
            f"epoch {epoch:02d} "
            f"train_loss={metrics.train_loss:.4f} "
            f"val_loss={metrics.val_loss:.4f} "
            f"rgb={metrics.val_rgb_l1:.4f} "
            f"chroma={metrics.val_chroma_l1:.4f} "
            f"blue={metrics.val_blue_penalty:.4f} "
            f"time={metrics.epoch_seconds/60.0:.1f}m"
        )

    write_summary(
        summary_path=output_dir / "summary.json",
        best_val_loss=best_val_loss,
        history=history,
        manifest_path=manifest_path,
        base_checkpoint_path=base_checkpoint_path,
    )
    log(f"Best checkpoint: {best_checkpoint_path}")
    log(f"Last checkpoint: {last_checkpoint_path}")
    log(f"State checkpoint: {training_state_path}")
    log(f"Metrics log:     {metrics_log_path}")
    return 0


class MovieColorDataset(Dataset):
    def __init__(self, samples: list[ManifestSample], image_size: int, train: bool):
        self.samples = samples
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.train:
            image = random_center_biased_square_crop(image, self.image_size)
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
        else:
            image = ImageOps.fit(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )

        target_rgb = TF.to_tensor(image)
        grayscale = image.convert("L").convert("RGB")
        input_rgb = TF.to_tensor(grayscale)
        return {
            "input_norm": normalize_image(input_rgb),
            "input_rgb": input_rgb,
            "target_rgb": target_rgb,
            "movie": sample.movie,
            "image_path": sample.image_path,
        }


def load_manifest(manifest_path: Path) -> list[ManifestSample]:
    samples: list[ManifestSample] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(
            ManifestSample(
                split=str(payload["split"]),
                movie=str(payload["movie"]),
                image_path=str(payload["image_path"]),
                sample_index=int(payload["sample_index"]),
                timestamp_seconds=float(payload["timestamp_seconds"]),
            )
        )
    return samples


def load_model(checkpoint_path: Path, device: torch.device) -> DeoldifyVideoModel:
    with torch.serialization.safe_globals([slice]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DeoldifyVideoModel()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    return model


def resolve_device(device_name: str) -> torch.device:
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this machine.")
        return torch.device("mps")
    return torch.device("cpu")


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(dtype=image.dtype)
    std = IMAGENET_STD.to(dtype=image.dtype)
    return (image - mean) / std


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image * std) + mean


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    train: bool,
) -> dict[str, float]:
    mode = "train" if train else "eval"
    getattr(model, mode)()
    totals = defaultdict(float)
    sample_count = 0
    for batch in dataloader:
        input_norm = batch["input_norm"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        prediction_norm = model(input_norm)
        prediction_rgb = denormalize_image(prediction_norm).clamp(0.0, 1.0)
        losses = compute_losses(prediction_rgb, target_rgb, args)
        loss = losses["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Encountered non-finite loss during training.")
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = input_norm.size(0)
        sample_count += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * batch_size
    if sample_count == 0:
        raise RuntimeError("Encountered an empty dataloader.")
    return {key: value / sample_count for key, value in totals.items()}


def compute_losses(
    prediction_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    center_loss_sigma = get_arg(args, "center_loss_sigma", 0.45)
    center_loss_strength = get_arg(args, "center_loss_strength", 2.0)
    blue_margin = get_arg(args, "blue_margin", 0.02)
    neutral_saturation_threshold = get_arg(args, "neutral_saturation_threshold", 0.10)
    rgb_loss_weight = get_arg(args, "rgb_loss_weight", 1.0)
    chroma_loss_weight = get_arg(args, "chroma_loss_weight", 1.10)
    saturation_loss_weight = get_arg(args, "saturation_loss_weight", 0.20)
    blue_penalty_weight = get_arg(args, "blue_penalty_weight", 0.08)
    neutral_chroma_penalty_weight = get_arg(args, "neutral_chroma_penalty_weight", 0.12)
    skin_blue_penalty_weight = get_arg(args, "skin_blue_penalty_weight", 0.35)
    costume_blue_penalty_weight = get_arg(args, "costume_blue_penalty_weight", 0.18)
    costume_dark_penalty_weight = get_arg(args, "costume_dark_penalty_weight", 0.20)
    costume_dark_margin = get_arg(args, "costume_dark_margin", 0.08)
    costume_vividness_penalty_weight = get_arg(args, "costume_vividness_penalty_weight", 0.28)
    costume_vividness_threshold = get_arg(args, "costume_vividness_threshold", 0.18)
    costume_vividness_margin = get_arg(args, "costume_vividness_margin", 0.03)

    prediction_luma = rgb_to_luma(prediction_rgb)
    target_luma = rgb_to_luma(target_rgb)
    prediction_chroma = prediction_rgb - prediction_luma
    target_chroma = target_rgb - target_luma
    prediction_saturation = torch.sqrt(torch.sum(prediction_chroma**2, dim=1, keepdim=True) + 1e-6)
    target_saturation = torch.sqrt(torch.sum(target_chroma**2, dim=1, keepdim=True) + 1e-6)
    center_prior = build_center_prior(
        height=target_rgb.shape[-2],
        width=target_rgb.shape[-1],
        device=target_rgb.device,
        dtype=target_rgb.dtype,
        sigma=center_loss_sigma,
    )
    center_weights = 1.0 + center_loss_strength * center_prior
    rgb_l1 = weighted_channel_mean(torch.abs(prediction_rgb - target_rgb), center_weights)
    chroma_l1 = weighted_channel_mean(torch.abs(prediction_chroma - target_chroma), center_weights)
    saturation_l1 = weighted_channel_mean(
        torch.abs(prediction_saturation - target_saturation),
        center_weights,
    )

    prediction_blue = prediction_rgb[:, 2:3]
    target_blue = target_rgb[:, 2:3]
    target_reference = torch.maximum(target_rgb[:, 0:1], target_rgb[:, 1:2])
    not_blue_target = torch.relu(target_reference - target_blue)
    excess_blue = torch.relu(prediction_blue - target_blue - blue_margin)
    blue_penalty = weighted_channel_mean(excess_blue * not_blue_target, center_weights)
    skin_mask = build_skin_mask(
        target_rgb=target_rgb,
        target_luma=target_luma,
        center_prior=center_prior,
    )
    costume_mask = build_costume_mask(
        target_luma=target_luma,
        target_saturation=target_saturation,
        center_prior=center_prior,
        skin_mask=skin_mask,
    )
    neutral_mask = build_neutral_mask(
        target_saturation=target_saturation,
        center_prior=center_prior,
        threshold=neutral_saturation_threshold,
    )
    skin_blue_penalty = weighted_channel_mean(excess_blue * not_blue_target, skin_mask)
    costume_blue_penalty = weighted_channel_mean(excess_blue * not_blue_target, costume_mask)
    costume_dark_penalty = weighted_channel_mean(
        torch.relu(target_luma - prediction_luma - costume_dark_margin),
        costume_mask * (target_luma > 0.18).to(target_luma.dtype),
    )
    neutral_chroma_penalty = weighted_channel_mean(prediction_saturation, neutral_mask)
    vivid_costume_mask = costume_mask * (target_saturation > costume_vividness_threshold).to(target_saturation.dtype)
    costume_vividness_penalty = weighted_channel_mean(
        torch.relu(target_saturation - prediction_saturation - costume_vividness_margin),
        vivid_costume_mask,
    )

    loss = (
        rgb_loss_weight * rgb_l1
        + chroma_loss_weight * chroma_l1
        + saturation_loss_weight * saturation_l1
        + blue_penalty_weight * blue_penalty
        + neutral_chroma_penalty_weight * neutral_chroma_penalty
        + skin_blue_penalty_weight * skin_blue_penalty
        + costume_blue_penalty_weight * costume_blue_penalty
        + costume_dark_penalty_weight * costume_dark_penalty
        + costume_vividness_penalty_weight * costume_vividness_penalty
    )
    return {
        "loss": loss,
        "rgb_l1": rgb_l1,
        "chroma_l1": chroma_l1,
        "saturation_l1": saturation_l1,
        "blue_penalty": blue_penalty,
        "skin_blue_penalty": skin_blue_penalty,
        "costume_blue_penalty": costume_blue_penalty,
        "costume_dark_penalty": costume_dark_penalty,
        "costume_vividness_penalty": costume_vividness_penalty,
        "neutral_chroma_penalty": neutral_chroma_penalty,
    }


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return torch.sum(rgb * weights, dim=1, keepdim=True)


def weighted_channel_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded_weights = weights.expand(values.shape[0], values.shape[1], values.shape[2], values.shape[3])
    return (values * expanded_weights).sum() / expanded_weights.sum().clamp_min(1e-6)


def get_arg(args: argparse.Namespace, name: str, default: float) -> float:
    value = getattr(args, name, default)
    return float(value)


def build_center_prior(
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float,
) -> torch.Tensor:
    y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    sigma_sq = max(sigma, 1e-3) ** 2
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma_sq))
    return gaussian.unsqueeze(0).unsqueeze(0)


def build_skin_mask(
    *,
    target_rgb: torch.Tensor,
    target_luma: torch.Tensor,
    center_prior: torch.Tensor,
) -> torch.Tensor:
    r_channel = target_rgb[:, 0:1]
    g_channel = target_rgb[:, 1:2]
    b_channel = target_rgb[:, 2:3]
    cb = -0.168736 * r_channel - 0.331264 * g_channel + 0.5 * b_channel + 0.5
    cr = 0.5 * r_channel - 0.418688 * g_channel - 0.081312 * b_channel + 0.5
    skin_mask = (
        (cb > 0.30)
        & (cb < 0.54)
        & (cr > 0.52)
        & (cr < 0.68)
        & (target_luma > 0.18)
        & (target_luma < 0.95)
        & (r_channel > g_channel * 0.85)
        & (r_channel > b_channel * 0.75)
    ).to(target_rgb.dtype)
    return skin_mask * torch.clamp(center_prior * 1.25, 0.0, 1.0)


def build_costume_mask(
    *,
    target_luma: torch.Tensor,
    target_saturation: torch.Tensor,
    center_prior: torch.Tensor,
    skin_mask: torch.Tensor,
) -> torch.Tensor:
    height = target_luma.shape[-2]
    y_coords = torch.linspace(0.0, 1.0, steps=height, device=target_luma.device, dtype=target_luma.dtype)
    lower_body_prior = y_coords.view(1, 1, height, 1)
    actor_prior = torch.clamp((center_prior - 0.15) / 0.85, 0.0, 1.0)
    chroma_or_fabric = torch.clamp(0.35 + target_saturation * 2.0, 0.0, 1.0)
    visibility = ((target_luma > 0.08) & (target_luma < 0.98)).to(target_luma.dtype)
    costume_mask = actor_prior * (0.65 + 0.35 * lower_body_prior) * chroma_or_fabric * visibility
    return costume_mask * (1.0 - skin_mask.clamp(0.0, 1.0))


def build_neutral_mask(
    *,
    target_saturation: torch.Tensor,
    center_prior: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    neutral_mask = (target_saturation < threshold).to(target_saturation.dtype)
    background_boost = 1.0 - torch.clamp(center_prior * 0.8, 0.0, 0.8)
    return neutral_mask * (0.35 + 0.65 * background_boost)


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: EpochMetrics,
) -> None:
    payload = {
        "model": model.state_dict(),
        "meta": {
            "epoch": epoch,
            "args": vars(args),
            "metrics": asdict(metrics),
        },
    }
    atomic_torch_save(payload, checkpoint_path)


def save_training_state(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
    history: list[EpochMetrics],
    base_checkpoint_path: Path,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "history": [asdict(item) for item in history],
        "base_checkpoint_path": str(base_checkpoint_path),
        "args": vars(args),
    }
    atomic_torch_save(payload, checkpoint_path)


def maybe_resume_state(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    training_state_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> ResumeState:
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
    elif args.resume:
        resume_path = training_state_path

    if resume_path is None:
        return ResumeState(
            start_epoch=1,
            best_val_loss=math.inf,
            history=[],
            base_checkpoint_path=checkpoint_path,
        )
    if not resume_path.exists():
        log(f"Resume state not found at {resume_path}. Starting fresh.")
        return ResumeState(
            start_epoch=1,
            best_val_loss=math.inf,
            history=[],
            base_checkpoint_path=checkpoint_path,
        )

    state = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    history = [EpochMetrics(**item) for item in state.get("history", [])]
    start_epoch = int(state["epoch"]) + 1
    best_val_loss = float(state.get("best_val_loss", math.inf))
    base_checkpoint_path = Path(state.get("base_checkpoint_path", checkpoint_path)).expanduser().resolve()
    log(f"Loaded resume state from {resume_path}")
    return ResumeState(
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        history=history,
        base_checkpoint_path=base_checkpoint_path,
    )


def write_summary(
    *,
    summary_path: Path,
    best_val_loss: float,
    history: list[EpochMetrics],
    manifest_path: Path,
    base_checkpoint_path: Path,
) -> None:
    summary_path.write_text(
        json.dumps(
            {
                "best_val_loss": best_val_loss,
                "epochs": [asdict(item) for item in history],
                "manifest": str(manifest_path),
                "base_checkpoint": str(base_checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def atomic_torch_save(payload: dict[str, object], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(checkpoint_path)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def render_preview(
    *,
    model: nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    output_path: Path,
    preview_count: int,
) -> None:
    model.eval()
    input_norm = batch["input_norm"][:preview_count].to(device)
    input_rgb = batch["input_rgb"][:preview_count]
    target_rgb = batch["target_rgb"][:preview_count]
    with torch.no_grad():
        prediction_rgb = denormalize_image(model(input_norm)).clamp(0.0, 1.0).cpu()
    rows: list[Image.Image] = []
    for source, prediction, target in zip(input_rgb, prediction_rgb, target_rgb, strict=True):
        row = stitch_triplet(
            tensor_to_image(source),
            tensor_to_image(prediction),
            tensor_to_image(target),
        )
        rows.append(row)
    preview = stack_rows(rows)
    preview.save(output_path)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def stitch_triplet(source: Image.Image, prediction: Image.Image, target: Image.Image) -> Image.Image:
    width, height = source.size
    canvas = Image.new("RGB", (width * 3, height))
    canvas.paste(source, (0, 0))
    canvas.paste(prediction, (width, 0))
    canvas.paste(target, (width * 2, 0))
    return canvas


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    if not rows:
        raise ValueError("No rows provided for preview rendering.")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height))
    y_offset = 0
    for row in rows:
        canvas.paste(row, (0, y_offset))
        y_offset += row.height
    return canvas


def random_center_biased_square_crop(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    min_side = min(width, height)
    crop_side = int(min_side * random.uniform(0.82, 1.0))
    max_left = max(width - crop_side, 0)
    max_top = max(height - crop_side, 0)
    center_left = max_left / 2.0
    center_top = max_top / 2.0
    jitter_left = max_left * 0.12
    jitter_top = max_top * 0.12
    left = int(round(min(max(random.uniform(center_left - jitter_left, center_left + jitter_left), 0.0), max_left)))
    top = int(round(min(max(random.uniform(center_top - jitter_top, center_top + jitter_top), 0.0), max_top)))
    return TF.resized_crop(
        image,
        top,
        left,
        crop_side,
        crop_side,
        size=[image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
