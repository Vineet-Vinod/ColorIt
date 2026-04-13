from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import denormalize_image, load_manifest, load_model, set_module_trainable
from scripts.finetune_deoldify_v1 import MovieColorDatasetV1, compute_losses


@dataclass(frozen=True)
class BenchConfig:
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class LossArgs:
    rgb_loss_weight: float = 1.0
    chroma_loss_weight: float = 1.0
    saturation_loss_weight: float = 0.15
    blue_penalty_weight: float = 0.0
    blue_margin: float = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the clean v1 full-model DeOldify fine-tune path.")
    parser.add_argument("--manifest", default="data/finetune/kannada_period/manifest.jsonl")
    parser.add_argument("--checkpoint", default="models/deoldify/ColorizeVideo_gen.pth")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("mps")
    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    samples = load_manifest(manifest_path)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    print(f"train_samples={len(train_samples)}")
    print(f"val_samples={len(val_samples)}")

    configs = [
        BenchConfig(batch_size=6, num_workers=0),
        BenchConfig(batch_size=8, num_workers=0),
        BenchConfig(batch_size=6, num_workers=4),
    ]

    for config in configs:
        try:
            result = run_benchmark(
                train_samples=train_samples,
                val_samples=val_samples,
                checkpoint_path=checkpoint_path,
                device=device,
                image_size=args.image_size,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                config=config,
            )
            print(result)
        except Exception as exc:
            print(
                {
                    "batch_size": config.batch_size,
                    "num_workers": config.num_workers,
                    "error": repr(exc),
                }
            )
            torch.mps.empty_cache()

    return 0


def run_benchmark(
    *,
    train_samples,
    val_samples,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
    warmup_steps: int,
    measure_steps: int,
    config: BenchConfig,
) -> dict[str, float | int]:
    torch.mps.empty_cache()
    train_dataset = MovieColorDatasetV1(train_samples, image_size=image_size, train=True)
    val_dataset = MovieColorDatasetV1(val_samples, image_size=image_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = load_model(checkpoint_path, device)
    set_module_trainable(model[0], True)
    model.train()

    encoder_parameters = list(model[0].parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 3e-5 * 0.05},
            {"params": decoder_parameters, "lr": 3e-5},
        ],
        weight_decay=1e-4,
    )

    loss_args = LossArgs()
    train_step_seconds = measure_train_steps(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        loss_args=loss_args,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )
    val_step_seconds = measure_val_steps(
        model=model,
        loader=val_loader,
        device=device,
        warmup_steps=max(2, warmup_steps // 2),
        measure_steps=max(10, measure_steps // 2),
    )

    train_images_per_s = config.batch_size / train_step_seconds
    val_images_per_s = config.batch_size / val_step_seconds
    train_epoch_minutes = len(train_samples) / train_images_per_s / 60.0
    val_epoch_minutes = len(val_samples) / val_images_per_s / 60.0
    epoch_minutes = train_epoch_minutes + val_epoch_minutes + 1.0

    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "train_step_s": round(train_step_seconds, 3),
        "train_images_per_s": round(train_images_per_s, 2),
        "val_step_s": round(val_step_seconds, 3),
        "val_images_per_s": round(val_images_per_s, 2),
        "est_epoch_min": round(epoch_minutes, 1),
        "est_4_epoch_h": round(epoch_minutes * 4 / 60.0, 2),
        "driver_mem_gb": round(torch.mps.driver_allocated_memory() / 1024**3, 2),
        "tensor_mem_gb": round(torch.mps.current_allocated_memory() / 1024**3, 2),
    }


def measure_train_steps(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_args: LossArgs,
    warmup_steps: int,
    measure_steps: int,
) -> float:
    timings: list[float] = []
    for step, batch in enumerate(loader, start=1):
        started = time.perf_counter()
        input_norm = batch["input_norm"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        prediction_norm = model(input_norm)
        prediction_rgb = denormalize_image(prediction_norm).clamp(0.0, 1.0)
        losses = compute_losses(prediction_rgb, target_rgb, loss_args)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        torch.mps.synchronize()
        timings.append(time.perf_counter() - started)
        if step >= warmup_steps + measure_steps:
            break
    measured = timings[warmup_steps:]
    return sum(measured) / len(measured)


def measure_val_steps(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> float:
    timings: list[float] = []
    model.eval()
    for step, batch in enumerate(loader, start=1):
        started = time.perf_counter()
        input_norm = batch["input_norm"].to(device)
        with torch.no_grad():
            _ = model(input_norm)
        torch.mps.synchronize()
        timings.append(time.perf_counter() - started)
        if step >= warmup_steps + measure_steps:
            break
    measured = timings[warmup_steps:]
    model.train()
    return sum(measured) / len(measured)


if __name__ == "__main__":
    raise SystemExit(main())
