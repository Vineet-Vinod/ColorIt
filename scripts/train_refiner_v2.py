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

from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import (
    append_jsonl,
    load_manifest,
    log,
    resolve_device,
    seed_everything,
)
from src.pipeline.refiner_arch import CostumeRefinerUNet
from src.pipeline.refiner_color import (
    apply_refiner_delta,
    compose_refiner_input,
    delta_ab_from_lab,
    grayscale_rgb_from_rgb_tensor,
    rgb_to_lab_tensor,
)


@dataclass(frozen=True)
class RefinerEpochMetrics:
    epoch: int
    train_loss: float
    train_delta_l1: float
    train_bg_identity: float
    train_skin_rgb: float
    train_skin_hue: float
    train_costume_rgb: float
    train_costume_hue: float
    train_costume_vividness: float
    val_loss: float
    val_delta_l1: float
    val_bg_identity: float
    val_skin_rgb: float
    val_skin_hue: float
    val_costume_rgb: float
    val_costume_hue: float
    val_costume_vividness: float
    epoch_seconds: float


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    best_val_loss: float
    history: list[RefinerEpochMetrics]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a post-DeOldify residual refiner for actor and costume colors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-root", required=True, help="Root directory containing cached base DeOldify outputs.")
    parser.add_argument("--mask-root", default=None, help="Optional person/skin/costume mask root.")
    parser.add_argument("--output-dir", default="models/refiner/v2")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--ab-delta-scale", type=float, default=24.0)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--focus-delta-boost", type=float, default=2.0)
    parser.add_argument("--global-rgb-weight", type=float, default=0.20)
    parser.add_argument("--background-identity-weight", type=float, default=0.55)
    parser.add_argument("--skin-rgb-weight", type=float, default=0.18)
    parser.add_argument("--skin-hue-weight", type=float, default=0.18)
    parser.add_argument("--costume-rgb-weight", type=float, default=0.24)
    parser.add_argument("--costume-hue-weight", type=float, default=0.18)
    parser.add_argument("--costume-vividness-weight", type=float, default=0.20)
    parser.add_argument("--costume-vividness-threshold", type=float, default=0.18)
    parser.add_argument("--actor-sampler-power", type=float, default=0.0)
    parser.add_argument("--actor-sampler-mask-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    seed_everything(args.seed)

    manifest_path = Path(args.manifest).expanduser().resolve()
    base_root = Path(args.base_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mask_root = Path(args.mask_root).expanduser().resolve() if args.mask_root else None
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(manifest_path)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    device = resolve_device(args.device)
    log(f"Device: {device}")
    log(f"Train samples: {len(train_samples)}")
    log(f"Val samples:   {len(val_samples)}")
    log(f"Base root:     {base_root}")
    log(f"Mask root:     {mask_root if mask_root else 'disabled'}")

    train_dataset = RefinerDataset(train_samples, image_size=args.image_size, train=True, base_root=base_root, mask_root=mask_root)
    val_dataset = RefinerDataset(val_samples, image_size=args.image_size, train=False, base_root=base_root, mask_root=mask_root)
    train_sampler = build_train_sampler(dataset=train_dataset, args=args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    model = CostumeRefinerUNet(input_channels=4, base_channels=args.base_channels, ab_delta_scale=args.ab_delta_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    metrics_log_path = output_dir / "metrics.jsonl"
    best_checkpoint_path = output_dir / "best.pth"
    last_checkpoint_path = output_dir / "last.pth"
    training_state_path = output_dir / "training_state.pth"
    epoch_checkpoint_dir = output_dir / "epoch_checkpoints"
    if args.save_every_epoch:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    resume_state = maybe_resume_state(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state_path=training_state_path,
        device=device,
    )
    best_val_loss = resume_state.best_val_loss
    history = resume_state.history
    preview_batch = next(iter(val_loader))

    for epoch in range(resume_state.start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(model=model, dataloader=train_loader, optimizer=optimizer, device=device, args=args, train=True)
        val_metrics = run_epoch(model=model, dataloader=val_loader, optimizer=None, device=device, args=args, train=False)
        scheduler.step()
        metrics = RefinerEpochMetrics(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_delta_l1=train_metrics["delta_l1"],
            train_bg_identity=train_metrics["bg_identity"],
            train_skin_rgb=train_metrics["skin_rgb"],
            train_skin_hue=train_metrics["skin_hue"],
            train_costume_rgb=train_metrics["costume_rgb"],
            train_costume_hue=train_metrics["costume_hue"],
            train_costume_vividness=train_metrics["costume_vividness"],
            val_loss=val_metrics["loss"],
            val_delta_l1=val_metrics["delta_l1"],
            val_bg_identity=val_metrics["bg_identity"],
            val_skin_rgb=val_metrics["skin_rgb"],
            val_skin_hue=val_metrics["skin_hue"],
            val_costume_rgb=val_metrics["costume_rgb"],
            val_costume_hue=val_metrics["costume_hue"],
            val_costume_vividness=val_metrics["costume_vividness"],
            epoch_seconds=time.perf_counter() - epoch_started,
        )
        history.append(metrics)
        append_jsonl(metrics_log_path, asdict(metrics))
        render_preview(model=model, batch=preview_batch, device=device, output_path=preview_dir / f"epoch_{epoch:02d}.png", preview_count=args.preview_count)
        save_checkpoint(last_checkpoint_path, model, args, epoch, metrics)
        if args.save_every_epoch:
            save_checkpoint(epoch_checkpoint_dir / f"epoch_{epoch:02d}.pth", model, args, epoch, metrics)
        save_training_state(training_state_path=training_state_path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_val_loss=best_val_loss, history=history)
        if metrics.val_loss < best_val_loss:
            best_val_loss = metrics.val_loss
            save_checkpoint(best_checkpoint_path, model, args, epoch, metrics)
            save_training_state(training_state_path=training_state_path, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_val_loss=best_val_loss, history=history)
        write_summary(output_dir / "summary.json", best_val_loss=best_val_loss, history=history, manifest_path=manifest_path, base_root=base_root)
        log(
            f"epoch {epoch:02d} train_loss={metrics.train_loss:.4f} val_loss={metrics.val_loss:.4f} "
            f"delta={metrics.val_delta_l1:.4f} bg={metrics.val_bg_identity:.4f} "
            f"skin_rgb={metrics.val_skin_rgb:.4f} skin_hue={metrics.val_skin_hue:.4f} "
            f"costume_rgb={metrics.val_costume_rgb:.4f} costume_hue={metrics.val_costume_hue:.4f} "
            f"time={metrics.epoch_seconds/60.0:.1f}m"
        )

    log(f"Best checkpoint: {best_checkpoint_path}")
    return 0


class RefinerDataset(Dataset):
    def __init__(self, samples: list, image_size: int, train: bool, base_root: Path, mask_root: Path | None):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.base_root = base_root
        self.mask_root = mask_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        target = Image.open(sample.image_path).convert("RGB")
        base = Image.open(self.base_root / sample.split / Path(sample.image_path).name).convert("RGB")
        person_mask = self._load_mask("person", sample)
        skin_mask = self._load_mask("skin", sample)
        costume_mask = self._load_mask("costume", sample)
        if self.train:
            target, base, person_mask, skin_mask, costume_mask = paired_random_center_biased_square_crop(
                target,
                base,
                self.image_size,
                person_mask,
                skin_mask,
                costume_mask,
            )
            if random.random() < 0.5:
                target = ImageOps.mirror(target)
                base = ImageOps.mirror(base)
                if person_mask is not None:
                    person_mask = ImageOps.mirror(person_mask)
                if skin_mask is not None:
                    skin_mask = ImageOps.mirror(skin_mask)
                if costume_mask is not None:
                    costume_mask = ImageOps.mirror(costume_mask)
        else:
            target = ImageOps.fit(target, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
            base = ImageOps.fit(base, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
            person_mask = fit_optional_mask(person_mask, self.image_size)
            skin_mask = fit_optional_mask(skin_mask, self.image_size)
            costume_mask = fit_optional_mask(costume_mask, self.image_size)

        target_rgb = TF.to_tensor(target)
        base_rgb = TF.to_tensor(base)
        gray_rgb = TF.to_tensor(target.convert("L").convert("RGB"))
        return {
            "base_rgb": base_rgb,
            "gray_rgb": gray_rgb,
            "target_rgb": target_rgb,
            "person_mask": mask_to_tensor(person_mask, self.image_size),
            "skin_mask": mask_to_tensor(skin_mask, self.image_size),
            "costume_mask": mask_to_tensor(costume_mask, self.image_size),
            "image_path": sample.image_path,
        }

    def _load_mask(self, mask_type: str, sample) -> Image.Image | None:
        if self.mask_root is None:
            return None
        path = self.mask_root / mask_type / sample.split / Path(sample.image_path).name
        if not path.exists():
            return None
        return Image.open(path).convert("L")

    def load_mask_mean(self, mask_type: str, index: int, size: int) -> float:
        sample = self.samples[index]
        mask = self._load_mask(mask_type, sample)
        if mask is None:
            return 0.0
        resized = mask.resize((size, size), Image.Resampling.BILINEAR)
        return float(TF.to_tensor(resized).mean().item())


def build_train_sampler(dataset: RefinerDataset, args: argparse.Namespace) -> WeightedRandomSampler | None:
    if (not dataset.train) or dataset.mask_root is None or float(args.actor_sampler_power) <= 0.0:
        return None
    mask_size = max(8, int(args.actor_sampler_mask_size))
    weights: list[float] = []
    for index in range(len(dataset)):
        person_mean = dataset.load_mask_mean("person", index, mask_size)
        skin_mean = dataset.load_mask_mean("skin", index, mask_size)
        costume_mean = dataset.load_mask_mean("costume", index, mask_size)
        actor_score = (0.45 * person_mean) + (0.60 * skin_mean) + (1.40 * costume_mean)
        weights.append(1.0 + float(args.actor_sampler_power) * actor_score)
    return WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True)


def fit_optional_mask(mask: Image.Image | None, image_size: int) -> Image.Image | None:
    if mask is None:
        return None
    return ImageOps.fit(mask, (image_size, image_size), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))


def paired_random_center_biased_square_crop(
    target: Image.Image,
    base: Image.Image,
    image_size: int,
    person_mask: Image.Image | None,
    skin_mask: Image.Image | None,
    costume_mask: Image.Image | None,
) -> tuple[Image.Image, Image.Image, Image.Image | None, Image.Image | None, Image.Image | None]:
    width, height = target.size
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
    target_crop = TF.resized_crop(target, top, left, crop_side, crop_side, size=[image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    base_crop = TF.resized_crop(base, top, left, crop_side, crop_side, size=[image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    person_crop = resize_mask_crop(person_mask, top, left, crop_side, image_size)
    skin_crop = resize_mask_crop(skin_mask, top, left, crop_side, image_size)
    costume_crop = resize_mask_crop(costume_mask, top, left, crop_side, image_size)
    return target_crop, base_crop, person_crop, skin_crop, costume_crop


def resize_mask_crop(mask: Image.Image | None, top: int, left: int, crop_side: int, image_size: int) -> Image.Image | None:
    if mask is None:
        return None
    return TF.resized_crop(mask, top, left, crop_side, crop_side, size=[image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)


def mask_to_tensor(mask: Image.Image | None, image_size: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros(1, image_size, image_size, dtype=torch.float32)
    return TF.to_tensor(mask).clamp(0.0, 1.0)


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
    train: bool,
) -> dict[str, float]:
    getattr(model, "train" if train else "eval")()
    totals = defaultdict(float)
    sample_count = 0
    for batch in dataloader:
        base_rgb = batch["base_rgb"].to(device)
        gray_rgb = batch["gray_rgb"].to(device)
        target_rgb = batch["target_rgb"].to(device)
        person_mask = batch["person_mask"].to(device)
        skin_mask = batch["skin_mask"].to(device)
        costume_mask = batch["costume_mask"].to(device)
        model_input = compose_refiner_input(base_rgb=base_rgb, gray_rgb=gray_rgb)
        predicted_delta_ab = model(model_input)
        losses = compute_losses(
            predicted_delta_ab=predicted_delta_ab,
            base_rgb=base_rgb,
            target_rgb=target_rgb,
            person_mask=person_mask,
            skin_mask=skin_mask,
            costume_mask=costume_mask,
            args=args,
        )
        loss = losses["loss"]
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = base_rgb.shape[0]
        sample_count += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * batch_size
    return {key: value / sample_count for key, value in totals.items()}


def compute_losses(
    *,
    predicted_delta_ab: torch.Tensor,
    base_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    person_mask: torch.Tensor,
    skin_mask: torch.Tensor,
    costume_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    base_lab = rgb_to_lab_tensor(base_rgb)
    target_lab = rgb_to_lab_tensor(target_rgb)
    target_delta_ab = delta_ab_from_lab(base_lab=base_lab, target_lab=target_lab)

    focus_mask = torch.maximum(person_mask, costume_mask).clamp(0.0, 1.0)
    has_focus = bool(torch.count_nonzero(focus_mask).item())
    if not has_focus:
        focus_mask = torch.ones_like(person_mask)
    background_mask = (1.0 - focus_mask).clamp(0.0, 1.0)

    refined_rgb = apply_refiner_delta(base_rgb=base_rgb, predicted_delta_ab=predicted_delta_ab)
    refined_lab = rgb_to_lab_tensor(refined_rgb)
    refined_delta_ab = delta_ab_from_lab(base_lab=base_lab, target_lab=refined_lab)

    focus_weights = 1.0 + float(args.focus_delta_boost) * focus_mask
    delta_l1 = weighted_mean(torch.abs(refined_delta_ab - target_delta_ab), focus_weights)
    global_rgb = torch.mean(torch.abs(refined_rgb - target_rgb))
    bg_identity = weighted_mean(torch.abs(predicted_delta_ab), background_mask)

    skin_rgb = torch.zeros((), device=base_rgb.device, dtype=base_rgb.dtype)
    if torch.count_nonzero(skin_mask).item() > 0:
        skin_rgb = weighted_mean(torch.abs(refined_rgb - target_rgb), skin_mask)

    skin_hue = torch.zeros((), device=base_rgb.device, dtype=base_rgb.dtype)
    if torch.count_nonzero(skin_mask).item() > 0:
        target_chroma = target_lab[:, 1:3]
        refined_chroma = refined_lab[:, 1:3]
        target_norm = torch.sqrt(torch.sum(target_chroma.square(), dim=1, keepdim=True) + 1e-6)
        refined_norm = torch.sqrt(torch.sum(refined_chroma.square(), dim=1, keepdim=True) + 1e-6)
        target_unit = target_chroma / target_norm.clamp_min(1e-4)
        refined_unit = refined_chroma / refined_norm.clamp_min(1e-4)
        alignment = 1.0 - torch.sum(target_unit * refined_unit, dim=1, keepdim=True).clamp(-1.0, 1.0)
        skin_hue = weighted_mean(alignment, skin_mask)

    costume_rgb = torch.zeros((), device=base_rgb.device, dtype=base_rgb.dtype)
    if torch.count_nonzero(costume_mask).item() > 0:
        costume_rgb = weighted_mean(torch.abs(refined_rgb - target_rgb), costume_mask)

    costume_hue = torch.zeros((), device=base_rgb.device, dtype=base_rgb.dtype)
    if torch.count_nonzero(costume_mask).item() > 0:
        target_chroma = target_lab[:, 1:3]
        refined_chroma = refined_lab[:, 1:3]
        target_sat = torch.sqrt(torch.sum(target_chroma.square(), dim=1, keepdim=True) + 1e-6)
        refined_sat = torch.sqrt(torch.sum(refined_chroma.square(), dim=1, keepdim=True) + 1e-6)
        target_unit = target_chroma / target_sat.clamp_min(1e-4)
        refined_unit = refined_chroma / refined_sat.clamp_min(1e-4)
        hue_alignment = 1.0 - torch.sum(target_unit * refined_unit, dim=1, keepdim=True).clamp(-1.0, 1.0)
        colorful_costume_mask = costume_mask * (target_sat > float(args.costume_vividness_threshold)).to(target_sat.dtype)
        costume_hue = weighted_mean(hue_alignment, colorful_costume_mask)

    costume_vividness = torch.zeros((), device=base_rgb.device, dtype=base_rgb.dtype)
    if torch.count_nonzero(costume_mask).item() > 0:
        target_sat = torch.sqrt(torch.sum(target_lab[:, 1:3].square(), dim=1, keepdim=True) + 1e-6)
        base_sat = torch.sqrt(torch.sum(base_lab[:, 1:3].square(), dim=1, keepdim=True) + 1e-6)
        refined_sat = torch.sqrt(torch.sum(refined_lab[:, 1:3].square(), dim=1, keepdim=True) + 1e-6)
        vivid_mask = costume_mask * (target_sat > float(args.costume_vividness_threshold)).to(target_sat.dtype)
        costume_vividness = weighted_mean(torch.relu(target_sat - refined_sat) * (target_sat > base_sat).to(target_sat.dtype), vivid_mask)

    loss = (
        float(args.delta_loss_weight) * delta_l1
        + float(args.global_rgb_weight) * global_rgb
        + float(args.background_identity_weight) * bg_identity
        + float(args.skin_rgb_weight) * skin_rgb
        + float(args.skin_hue_weight) * skin_hue
        + float(args.costume_rgb_weight) * costume_rgb
        + float(args.costume_hue_weight) * costume_hue
        + float(args.costume_vividness_weight) * costume_vividness
    )
    return {
        "loss": loss,
        "delta_l1": delta_l1,
        "bg_identity": bg_identity,
        "skin_rgb": skin_rgb,
        "skin_hue": skin_hue,
        "costume_rgb": costume_rgb,
        "costume_hue": costume_hue,
        "costume_vividness": costume_vividness,
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded = weights.expand(values.shape[0], values.shape[1], values.shape[2], values.shape[3])
    return (values * expanded).sum() / expanded.sum().clamp_min(1e-6)


def render_preview(*, model: nn.Module, batch: dict[str, torch.Tensor | list[str]], device: torch.device, output_path: Path, preview_count: int) -> None:
    model.eval()
    base_rgb = batch["base_rgb"][:preview_count].to(device)
    gray_rgb = batch["gray_rgb"][:preview_count].to(device)
    target_rgb = batch["target_rgb"][:preview_count]
    with torch.no_grad():
        predicted = model(compose_refiner_input(base_rgb=base_rgb, gray_rgb=gray_rgb))
        refined_rgb = apply_refiner_delta(base_rgb=base_rgb, predicted_delta_ab=predicted).cpu()
    rows = []
    for gray, base, refined, target in zip(gray_rgb.cpu(), base_rgb.cpu(), refined_rgb, target_rgb, strict=True):
        rows.append(stitch_quad(gray, base, refined, target))
    stack_rows(rows).save(output_path)


def stitch_quad(gray: torch.Tensor, base: torch.Tensor, refined: torch.Tensor, target: torch.Tensor) -> Image.Image:
    images = [tensor_to_image(gray), tensor_to_image(base), tensor_to_image(refined), tensor_to_image(target)]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 4, height))
    for idx, image in enumerate(images):
        canvas.paste(image, (idx * width, 0))
    return canvas


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0.0, 255.0).astype("uint8")
    return Image.fromarray(array)


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    return canvas


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, epoch: int, metrics: RefinerEpochMetrics) -> None:
    payload = {
        "model": model.state_dict(),
        "meta": {
            "epoch": epoch,
            "metrics": asdict(metrics),
            "model_args": {
                "input_channels": 4,
                "base_channels": int(args.base_channels),
                "ab_delta_scale": float(args.ab_delta_scale),
            },
            "args": vars(args),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def save_training_state(*, training_state_path: Path, model: nn.Module, optimizer, scheduler, epoch: int, best_val_loss: float, history: list[RefinerEpochMetrics]) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "history": [asdict(item) for item in history],
    }
    training_state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = training_state_path.with_suffix(training_state_path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(training_state_path)


def maybe_resume_state(*, args: argparse.Namespace, model: nn.Module, optimizer, scheduler, training_state_path: Path, device: torch.device) -> ResumeState:
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
    elif args.resume:
        resume_path = training_state_path
    if resume_path is None or not resume_path.exists():
        return ResumeState(start_epoch=1, best_val_loss=math.inf, history=[])
    state = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    history = [RefinerEpochMetrics(**item) for item in state.get("history", [])]
    return ResumeState(start_epoch=int(state["epoch"]) + 1, best_val_loss=float(state.get("best_val_loss", math.inf)), history=history)


def write_summary(path: Path, *, best_val_loss: float, history: list[RefinerEpochMetrics], manifest_path: Path, base_root: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "best_val_loss": best_val_loss,
                "epochs": [asdict(item) for item in history],
                "manifest": str(manifest_path),
                "base_root": str(base_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
