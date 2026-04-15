from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_deoldify import load_manifest, log
from src.pipeline.config import load_config
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.inference import colorize_pil_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen DeOldify outputs for refiner training.")
    parser.add_argument("--manifest", default="data/finetune/kannada_period/manifest.jsonl")
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--checkpoint", default="models/deoldify/ColorizeVideo_gen.pth")
    parser.add_argument("--output-root", default="data/finetune/refiner_cache/kannada_period")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(manifest_path)
    config = load_config(config_path)
    config.raw["model"]["weights_path"] = str(Path(args.checkpoint).expanduser().resolve())
    bundle = load_colorizer_bundle(config)
    render_factor = int(config.model["render_factor"])

    written = 0
    for index, sample in enumerate(samples, start=1):
        target_path = Path(sample.image_path).expanduser().resolve()
        output_path = output_root / sample.split / target_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            continue
        target = Image.open(target_path).convert("RGB")
        gray = target.convert("L").convert("RGB")
        base = colorize_pil_image(
            model_bundle=bundle,
            input_image=gray,
            render_factor=render_factor,
            postprocess_config=config.raw.get("postprocess", {}),
        )
        base.save(output_path)
        written += 1
        if index % 250 == 0:
            log(f"cached {index}/{len(samples)} frames")

    summary = {
        "manifest": str(manifest_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "output_root": str(output_root),
        "samples": len(samples),
        "written": written,
        "render_factor": render_factor,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Cache ready at {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
