from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import load_config
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.inference import colorize_pil_image
from src.pipeline.refiner_runtime import load_refiner_bundle, refine_base_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render still comparisons for base DeOldify versus the v2 refiner.")
    parser.add_argument("--movie", required=True)
    parser.add_argument("--timestamp", action="append", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_ROOT
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    movie = Path(args.movie).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    config = load_config(config_path)
    base_bundle = load_colorizer_bundle(config)
    refiner_bundle = load_refiner_bundle(checkpoint_path=Path(args.refiner_checkpoint), device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    font = ImageFont.load_default()
    rows = []

    for timestamp in args.timestamp:
        safe = timestamp.replace(":", "")
        orig_path = output_root / f"original_{safe}.png"
        bw_path = output_root / f"bw_{safe}.png"
        base_path = output_root / f"base_{safe}.png"
        refined_path = output_root / f"refined_{safe}.png"
        subprocess.run(["ffmpeg", "-y", "-ss", timestamp, "-i", str(movie), "-frames:v", "1", "-update", "1", str(orig_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        original = Image.open(orig_path).convert("RGB")
        gray = original.convert("L").convert("RGB")
        gray.save(bw_path)
        base = colorize_pil_image(model_bundle=base_bundle, input_image=gray, render_factor=int(config.model["render_factor"]), postprocess_config=config.raw.get("postprocess", {}))
        base.save(base_path)
        refined = refine_base_pil(bundle=refiner_bundle, base_image=base, gray_image=gray)
        refined.save(refined_path)
        row = Image.new("RGB", (original.width * 4, original.height + 28), (20, 20, 20))
        draw = ImageDraw.Draw(row)
        labels = ["Original", "B/W", "Base", "Refined"]
        for index, (label, image) in enumerate(zip(labels, [original, gray, base, refined], strict=True)):
            x = index * original.width
            row.paste(image, (x, 28))
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (original.width - tw) // 2, 8), label, fill=(245, 245, 245), font=font)
        rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), (8, 8, 8))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet_path = output_root / "comparison_sheet.png"
    sheet.save(sheet_path)
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
