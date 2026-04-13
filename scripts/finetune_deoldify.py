from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
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
        default=1.75,
        help="Weight for chroma reconstruction loss.",
    )
    parser.add_argument(
        "--saturation-loss-weight",
        type=float,
        default=0.5,
        help="Weight for saturation magnitude matching.",
    )
    parser.add_argument(
        "--blue-penalty-weight",
        type=float,
        default=0.35,
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
    return parser.parse_args()


def main() -> int:
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
    print(f"Device: {device}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples:   {len(val_samples)}")

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
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_val_loss = math.inf
    preview_batch = next(iter(val_loader))
    history: list[EpochMetrics] = []
    for epoch in range(1, args.epochs + 1):
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
        if metrics.val_loss < best_val_loss:
            best_val_loss = metrics.val_loss
            save_checkpoint(best_checkpoint_path, model, args, epoch, metrics)
        print(
            f"epoch {epoch:02d} "
            f"train_loss={metrics.train_loss:.4f} "
            f"val_loss={metrics.val_loss:.4f} "
            f"rgb={metrics.val_rgb_l1:.4f} "
            f"chroma={metrics.val_chroma_l1:.4f} "
            f"blue={metrics.val_blue_penalty:.4f} "
            f"time={metrics.epoch_seconds/60.0:.1f}m"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_val_loss": best_val_loss,
                "epochs": [asdict(item) for item in history],
                "manifest": str(manifest_path),
                "base_checkpoint": str(checkpoint_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Metrics log:     {metrics_log_path}")
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
            image = random_resized_square_crop(image, self.image_size)
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
    rgb_l1 = torch.mean(torch.abs(prediction_rgb - target_rgb))

    prediction_luma = rgb_to_luma(prediction_rgb)
    target_luma = rgb_to_luma(target_rgb)
    prediction_chroma = prediction_rgb - prediction_luma
    target_chroma = target_rgb - target_luma
    chroma_l1 = torch.mean(torch.abs(prediction_chroma - target_chroma))

    prediction_saturation = torch.sqrt(torch.sum(prediction_chroma**2, dim=1, keepdim=True) + 1e-6)
    target_saturation = torch.sqrt(torch.sum(target_chroma**2, dim=1, keepdim=True) + 1e-6)
    saturation_l1 = torch.mean(torch.abs(prediction_saturation - target_saturation))

    prediction_blue = prediction_rgb[:, 2:3]
    target_blue = target_rgb[:, 2:3]
    target_reference = torch.maximum(target_rgb[:, 0:1], target_rgb[:, 1:2])
    not_blue_target = torch.relu(target_reference - target_blue)
    excess_blue = torch.relu(prediction_blue - target_blue - args.blue_margin)
    blue_penalty = torch.mean(excess_blue * not_blue_target)

    loss = (
        args.rgb_loss_weight * rgb_l1
        + args.chroma_loss_weight * chroma_l1
        + args.saturation_loss_weight * saturation_l1
        + args.blue_penalty_weight * blue_penalty
    )
    return {
        "loss": loss,
        "rgb_l1": rgb_l1,
        "chroma_l1": chroma_l1,
        "saturation_l1": saturation_l1,
        "blue_penalty": blue_penalty,
    }


def rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return torch.sum(rgb * weights, dim=1, keepdim=True)


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
    torch.save(payload, checkpoint_path)


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


def random_resized_square_crop(image: Image.Image, image_size: int) -> Image.Image:
    i, j, h, w = RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.9, 1.1))
    return TF.resized_crop(
        image,
        i,
        j,
        h,
        w,
        size=[image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
