from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import (
    EpochMetrics,
    ManifestSample,
    append_jsonl,
    denormalize_image,
    load_manifest,
    load_model,
    log,
    maybe_resume_state,
    normalize_image,
    random_center_biased_square_crop,
    render_preview,
    resolve_device,
    save_checkpoint,
    save_training_state,
    seed_everything,
    set_module_trainable,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeOldify v1 full-model fine-tuning without masks or heuristic spatial losses."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl from prepare_finetune_dataset.py.")
    parser.add_argument(
        "--checkpoint",
        default="models/deoldify/ColorizeVideo_gen.pth",
        help="Base DeOldify generator checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/deoldify/finetune/v1_fullmodel",
        help="Directory for checkpoints, metrics, and previews.",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of fine-tuning epochs.")
    parser.add_argument("--image-size", type=int, default=256, help="Square crop size used during training.")
    parser.add_argument("--batch-size", type=int, default=6, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Decoder learning rate.")
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.05,
        help="Multiplier applied to the encoder learning rate.",
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
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
        default=0,
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
        default=1.0,
        help="Weight for chroma reconstruction loss.",
    )
    parser.add_argument(
        "--saturation-loss-weight",
        type=float,
        default=0.15,
        help="Weight for saturation magnitude matching.",
    )
    parser.add_argument(
        "--blue-penalty-weight",
        type=float,
        default=0.0,
        help="Optional global blue-overprediction penalty weight. Defaults to disabled.",
    )
    parser.add_argument(
        "--blue-margin",
        type=float,
        default=0.02,
        help="Tolerance before the diagnostic blue metric activates.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="How many validation samples to render into each epoch preview image.",
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
    log("Mask root:     disabled")

    train_dataset = MovieColorDatasetV1(train_samples, image_size=args.image_size, train=True)
    val_dataset = MovieColorDatasetV1(val_samples, image_size=args.image_size, train=False)

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
    config_path.write_text(__import__("json").dumps(vars(args), indent=2), encoding="utf-8")
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
        append_jsonl(metrics_log_path, __import__("dataclasses").asdict(metrics))
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


class MovieColorDatasetV1(Dataset):
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
            image, _, _, _ = random_center_biased_square_crop(image, self.image_size)
            if torch.rand(1).item() < 0.5:
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
    rgb_l1 = torch.mean(torch.abs(prediction_rgb - target_rgb))

    prediction_luma = rgb_to_luma(prediction_rgb)
    target_luma = rgb_to_luma(target_rgb)
    prediction_chroma = prediction_rgb - prediction_luma
    target_chroma = target_rgb - target_luma
    chroma_l1 = torch.mean(torch.abs(prediction_chroma - target_chroma))

    prediction_saturation = torch.sqrt(torch.sum(prediction_chroma.square(), dim=1, keepdim=True) + 1e-6)
    target_saturation = torch.sqrt(torch.sum(target_chroma.square(), dim=1, keepdim=True) + 1e-6)
    saturation_l1 = torch.mean(torch.abs(prediction_saturation - target_saturation))

    prediction_blue = prediction_rgb[:, 2:3]
    target_blue = target_rgb[:, 2:3]
    target_reference = torch.maximum(target_rgb[:, 0:1], target_rgb[:, 1:2])
    not_blue_target = torch.relu(target_reference - target_blue)
    excess_blue = torch.relu(prediction_blue - target_blue - float(args.blue_margin))
    blue_penalty = torch.mean(excess_blue * not_blue_target)

    loss = (
        float(args.rgb_loss_weight) * rgb_l1
        + float(args.chroma_loss_weight) * chroma_l1
        + float(args.saturation_loss_weight) * saturation_l1
        + float(args.blue_penalty_weight) * blue_penalty
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


if __name__ == "__main__":
    raise SystemExit(main())
