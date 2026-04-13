from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import load_manifest


PERSON_CLASS_INDEX = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate person, skin, and costume masks for fine-tuning.")
    parser.add_argument("--manifest", required=True, help="Path to the fine-tune manifest.jsonl.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for masks. Defaults to <manifest_dir>/masks.",
    )
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--person-threshold", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--debug-count", type=int, default=12, help="Number of debug overlays to save.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = resolve_output_root(args.output_root, manifest_path)
    if output_root.exists() and args.overwrite:
        for subdir in ("person", "skin", "costume", "debug"):
            target = output_root / subdir
            if target.exists():
                for path in sorted(target.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = deeplabv3_resnet50(weights=weights).to(device).eval()

    samples = load_manifest(manifest_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"device={device}")
    print(f"samples={len(samples)}")
    print(f"output_root={output_root}")

    debug_saved = 0
    summary = {"samples": 0, "with_person": 0, "with_skin": 0, "with_costume": 0}
    for batch_start in range(0, len(samples), args.batch_size):
        batch_samples = samples[batch_start : batch_start + args.batch_size]
        batch_images = [Image.open(sample.image_path).convert("RGB") for sample in batch_samples]
        batch_tensors = torch.stack([preprocess(image) for image in batch_images]).to(device)
        with torch.no_grad():
            output = model(batch_tensors)["out"]
            probabilities = torch.softmax(output, dim=1)

        for sample, image, probs in zip(batch_samples, batch_images, probabilities, strict=True):
            image_np = np.asarray(image)
            person_prob = probs[PERSON_CLASS_INDEX : PERSON_CLASS_INDEX + 1]
            person_prob = F.interpolate(
                person_prob.unsqueeze(0),
                size=image_np.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0].detach().cpu().numpy()
            person_mask = select_primary_person_mask(person_prob, args.person_threshold)
            skin_mask = derive_skin_mask(image_np, person_mask)
            costume_mask = derive_costume_mask(person_mask, skin_mask)
            write_mask(output_root / "person" / sample.split / Path(sample.image_path).name, person_mask)
            write_mask(output_root / "skin" / sample.split / Path(sample.image_path).name, skin_mask)
            write_mask(output_root / "costume" / sample.split / Path(sample.image_path).name, costume_mask)
            if debug_saved < args.debug_count:
                write_debug_overlay(
                    output_root / "debug" / sample.split / f"{Path(sample.image_path).stem}_overlay.png",
                    image_np,
                    person_mask,
                    skin_mask,
                    costume_mask,
                )
                debug_saved += 1
            summary["samples"] += 1
            summary["with_person"] += int(person_mask.any())
            summary["with_skin"] += int(skin_mask.any())
            summary["with_costume"] += int(costume_mask.any())

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


def resolve_output_root(output_root_arg: str | None, manifest_path: Path) -> Path:
    if output_root_arg:
        return Path(output_root_arg).expanduser().resolve()
    return (manifest_path.parent / "masks").resolve()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_primary_person_mask(person_prob: np.ndarray, threshold: float) -> np.ndarray:
    binary = (person_prob >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        adaptive_threshold = max(float(person_prob.max()) * 0.55, 0.18)
        binary = (person_prob >= adaptive_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    components = cv2.connectedComponentsWithStats(binary, connectivity=8)
    num_labels, labels, stats, _ = components
    if num_labels <= 1:
        return postprocess_binary_mask(binary)

    center_prior = build_center_prior(binary.shape[0], binary.shape[1])
    best_label = 1
    best_score = -1.0
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        area = float(stats[label, cv2.CC_STAT_AREA])
        score = float((component * person_prob * center_prior).sum()) + area * 1e-4
        if score > best_score:
            best_score = score
            best_label = label
    primary = (labels == best_label).astype(np.uint8)
    return postprocess_binary_mask(primary)


def derive_skin_mask(image_rgb: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    if not person_mask.any():
        return np.zeros_like(person_mask, dtype=np.uint8)

    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:, :, 0]
    cr_channel = ycrcb[:, :, 1]
    cb_channel = ycrcb[:, :, 2]
    r_channel = image_rgb[:, :, 0]
    g_channel = image_rgb[:, :, 1]
    b_channel = image_rgb[:, :, 2]

    skin = (
        (cr_channel >= 128)
        & (cr_channel <= 185)
        & (cb_channel >= 85)
        & (cb_channel <= 140)
        & (y_channel >= 40)
        & (r_channel >= g_channel * 0.9)
        & (r_channel >= b_channel * 0.8)
    )

    upper_body_prior = np.linspace(1.0, 0.25, image_rgb.shape[0], dtype=np.float32)[:, None]
    center_prior = build_center_prior(image_rgb.shape[0], image_rgb.shape[1])
    weighted_skin = skin.astype(np.float32) * upper_body_prior * np.clip(center_prior * 1.4, 0.0, 1.0)
    skin_mask = (weighted_skin > 0.2).astype(np.uint8) * person_mask
    kernel = make_kernel(image_rgb.shape[0], image_rgb.shape[1], scale=0.01)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    return skin_mask.astype(np.uint8)


def derive_costume_mask(person_mask: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    if not person_mask.any():
        return np.zeros_like(person_mask, dtype=np.uint8)
    kernel = make_kernel(person_mask.shape[0], person_mask.shape[1], scale=0.012)
    expanded_skin = cv2.dilate(skin_mask, kernel, iterations=1)
    costume_mask = person_mask.copy()
    costume_mask[expanded_skin > 0] = 0
    costume_mask = cv2.morphologyEx(costume_mask, cv2.MORPH_OPEN, kernel)
    costume_mask = cv2.morphologyEx(costume_mask, cv2.MORPH_CLOSE, kernel)
    return costume_mask.astype(np.uint8)


def postprocess_binary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = make_kernel(mask.shape[0], mask.shape[1], scale=0.012)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned.astype(np.uint8)


def make_kernel(height: int, width: int, scale: float) -> np.ndarray:
    size = max(3, int(round(min(height, width) * scale)))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_center_prior(height: int, width: int) -> np.ndarray:
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2.0 * 0.45**2))


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def write_debug_overlay(
    path: Path,
    image_rgb: np.ndarray,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
    costume_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_rgb.astype(np.float32).copy()
    overlay[person_mask > 0] = overlay[person_mask > 0] * 0.65 + np.array([40, 120, 255], dtype=np.float32) * 0.35
    overlay[skin_mask > 0] = overlay[skin_mask > 0] * 0.5 + np.array([255, 140, 80], dtype=np.float32) * 0.5
    overlay[costume_mask > 0] = overlay[costume_mask > 0] * 0.5 + np.array([60, 220, 120], dtype=np.float32) * 0.5
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path)


if __name__ == "__main__":
    raise SystemExit(main())
