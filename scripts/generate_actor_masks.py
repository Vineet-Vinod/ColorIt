from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from src.pipeline.actor_masks import (
    derive_costume_mask,
    derive_skin_mask,
    select_primary_person_mask,
    write_debug_overlay,
    write_mask,
)

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

if __name__ == "__main__":
    raise SystemExit(main())
