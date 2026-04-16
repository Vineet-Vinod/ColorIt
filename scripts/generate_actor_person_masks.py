from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import load_manifest
from src.pipeline.actor_mask_backends import (
    list_actor_mask_backends,
    load_actor_mask_backend,
    predict_actor_mask,
    resolve_device,
    write_actor_overlay,
    write_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conservative actor union masks from a fine-tune manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the fine-tune manifest.jsonl.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for masks. Defaults to <manifest_dir>/actor_masks.",
    )
    parser.add_argument(
        "--backend",
        choices=list_actor_mask_backends(),
        default="maskrcnn_v2_conservative",
        help="Actor-mask backend.",
    )
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--debug-count", type=int, default=12, help="Number of overlay images to save.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = resolve_output_root(args.output_root, manifest_path)
    if output_root.exists() and args.overwrite:
        for subdir in ("person", "debug"):
            target = output_root / subdir
            if target.exists():
                for path in sorted(target.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    backend = load_actor_mask_backend(args.backend, device=device)
    samples = load_manifest(manifest_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    print(f"device={device}")
    print(f"backend={args.backend}")
    print(f"samples={len(samples)}")
    print(f"output_root={output_root}")

    debug_saved = 0
    summary = {
        "samples": 0,
        "with_person": 0,
        "backend": args.backend,
        "device": str(device),
        "mean_mask_fraction": 0.0,
        "mean_inference_seconds": 0.0,
        "frames": [],
    }
    total_mask_fraction = 0.0
    total_inference_seconds = 0.0

    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        image_np = np.asarray(image)
        result = predict_actor_mask(backend, image)
        write_mask(output_root / "person" / sample.split / Path(sample.image_path).name, result.mask)
        if debug_saved < args.debug_count:
            write_actor_overlay(
                output_root / "debug" / sample.split / f"{Path(sample.image_path).stem}_overlay.png",
                image_np,
                result.mask,
                label=f"{args.backend} {result.metadata['mask_fraction']:.3f}",
            )
            debug_saved += 1

        summary["samples"] += 1
        summary["with_person"] += int(result.mask.any())
        total_mask_fraction += float(result.metadata["mask_fraction"])
        total_inference_seconds += float(result.metadata["inference_seconds"])
        summary["frames"].append(
            {
                "image_path": sample.image_path,
                "split": sample.split,
                "mask_fraction": float(result.metadata["mask_fraction"]),
                "inference_seconds": float(result.metadata["inference_seconds"]),
                "metadata": result.metadata,
            }
        )

    if summary["samples"] > 0:
        summary["mean_mask_fraction"] = total_mask_fraction / summary["samples"]
        summary["mean_inference_seconds"] = total_inference_seconds / summary["samples"]

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


def resolve_output_root(output_root_arg: str | None, manifest_path: Path) -> Path:
    if output_root_arg:
        return Path(output_root_arg).expanduser().resolve()
    return (manifest_path.parent / "actor_masks").resolve()


if __name__ == "__main__":
    raise SystemExit(main())
