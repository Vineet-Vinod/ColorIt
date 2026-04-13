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

from scripts.finetune_deoldify import (
    MovieColorDataset,
    compute_losses,
    denormalize_image,
    load_manifest,
    load_model,
    set_module_trainable,
)


@dataclass(frozen=True)
class BenchConfig:
    batch_size: int
    num_workers: int
    train_encoder: bool


@dataclass(frozen=True)
class LossArgs:
    blue_margin: float = 0.02
    rgb_loss_weight: float = 1.0
    chroma_loss_weight: float = 1.75
    saturation_loss_weight: float = 0.5
    blue_penalty_weight: float = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeOldify fine-tuning throughput on the local dataset.")
    parser.add_argument("--manifest", default="data/finetune/kannada_period/manifest.jsonl")
    parser.add_argument("--checkpoint", default="models/deoldify/ColorizeVideo_gen.pth")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("mps")
    manifest_path = Path(args.manifest).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    samples = load_manifest(manifest_path)
    train_samples = [sample for sample in samples if sample.split == "train"]
    print(f"train_samples={len(train_samples)}")

    base_model = load_model(checkpoint_path, device)
    total_params = sum(parameter.numel() for parameter in base_model.parameters())
    encoder_params = sum(parameter.numel() for parameter in base_model[0].parameters())
    decoder_params = total_params - encoder_params
    print(
        {
            "total_params_m": round(total_params / 1_000_000, 2),
            "encoder_params_m": round(encoder_params / 1_000_000, 2),
            "decoder_params_m": round(decoder_params / 1_000_000, 2),
        }
    )
    del base_model
    torch.mps.empty_cache()

    configs = [
        BenchConfig(batch_size=4, num_workers=0, train_encoder=False),
        BenchConfig(batch_size=4, num_workers=4, train_encoder=False),
        BenchConfig(batch_size=6, num_workers=4, train_encoder=False),
        BenchConfig(batch_size=8, num_workers=4, train_encoder=False),
        BenchConfig(batch_size=4, num_workers=4, train_encoder=True),
        BenchConfig(batch_size=6, num_workers=4, train_encoder=True),
        BenchConfig(batch_size=8, num_workers=4, train_encoder=True),
    ]

    for config in configs:
        try:
            result = run_benchmark(
                train_samples=train_samples,
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
                    "train_encoder": config.train_encoder,
                    "error": repr(exc),
                }
            )
            torch.mps.empty_cache()

    return 0


def run_benchmark(
    *,
    train_samples,
    checkpoint_path: Path,
    device: torch.device,
    image_size: int,
    warmup_steps: int,
    measure_steps: int,
    config: BenchConfig,
) -> dict[str, float | int | bool]:
    torch.mps.empty_cache()
    dataset = MovieColorDataset(train_samples, image_size=image_size, train=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    model = load_model(checkpoint_path, device)
    set_module_trainable(model[0], config.train_encoder)
    model.train()

    encoder_parameters = list(model[0].parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 1e-4 * 0.25},
            {"params": decoder_parameters, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    timings: list[float] = []
    waits: list[float] = []
    loss_args = LossArgs()
    start_wait = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        waits.append(time.perf_counter() - start_wait)
        step_started = time.perf_counter()

        input_norm = batch["input_norm"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        prediction_norm = model(input_norm)
        prediction_rgb = denormalize_image(prediction_norm).clamp(0.0, 1.0)
        losses = compute_losses(prediction_rgb, target_rgb, loss_args)
        loss = losses["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()

        timings.append(time.perf_counter() - step_started)
        if step >= warmup_steps + measure_steps:
            break
        start_wait = time.perf_counter()

    measured_timings = timings[warmup_steps:]
    measured_waits = waits[warmup_steps:]
    avg_step_seconds = sum(measured_timings) / len(measured_timings)
    avg_wait_seconds = sum(measured_waits) / len(measured_waits)
    images_per_second = config.batch_size / avg_step_seconds

    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "train_encoder": config.train_encoder,
        "avg_step_s": round(avg_step_seconds, 3),
        "images_per_s": round(images_per_second, 2),
        "data_wait_s": round(avg_wait_seconds, 3),
        "driver_mem_gb": round(torch.mps.driver_allocated_memory() / 1024**3, 2),
        "tensor_mem_gb": round(torch.mps.current_allocated_memory() / 1024**3, 2),
        "recommended_max_gb": round(torch.mps.recommended_max_memory() / 1024**3, 2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
