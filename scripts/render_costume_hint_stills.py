from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.config import load_config
from src.pipeline.costume_hint_recolor import recolor_costume_with_hints
from src.pipeline.costume_palette import build_costume_palette_runtime
from src.pipeline.ffmpeg_utils import extract_single_frame
from src.pipeline.inference import colorize_pil_image
from src.pipeline.probes import parse_timecode_to_seconds
from src.pipeline.model_loader import load_colorizer_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render DeOldify base frames and costume-only hint recolor comparisons."
    )
    parser.add_argument("--movie", required=True)
    parser.add_argument("--timestamp", action="append", required=True)
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--reference-image", action="append", default=[])
    parser.add_argument("--palette-color", action="append", default=[])
    parser.add_argument("--mask-backend", default="maskrcnn_v2_conservative")
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    movie = Path(args.movie).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    bundle = load_colorizer_bundle(config)
    font = ImageFont.load_default()

    frame_dir = output_root / "frames"
    output_dir = output_root / "outputs"
    sheet_dir = output_root / "sheets"
    for directory in (frame_dir, output_dir, sheet_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runtime = build_runtime_from_args(config=config, args=args, device=device)
    summary: dict[str, object] = {
        "movie": str(movie),
        "config": str(config_path),
        "palette_colors": list(args.palette_color),
        "reference_images": [str(Path(path).expanduser().resolve()) for path in args.reference_image],
        "timestamps": [],
    }
    sheets: list[Image.Image] = []

    for timestamp in args.timestamp:
        safe = timestamp.replace(":", "_").replace(".", "_")
        timestamp_dir = output_dir / safe
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        original_path = frame_dir / f"{safe}_original.png"
        bw_path = timestamp_dir / "bw.png"
        base_path = timestamp_dir / "base.png"
        direct_path = timestamp_dir / "direct_palette.png"
        hint_path = timestamp_dir / "hint_recolor.png"
        seeds_path = timestamp_dir / "seed_overlay.png"
        mask_path = timestamp_dir / "mask_overlay.png"
        costume_mask_path = timestamp_dir / "costume_mask.png"

        extract_single_frame(
            input_path=movie,
            output_path=original_path,
            time_seconds=parse_timecode_to_seconds(timestamp),
        )
        original = Image.open(original_path).convert("RGB")
        gray = original.convert("L").convert("RGB")
        gray.save(bw_path)
        base = colorize_pil_image(
            model_bundle=bundle,
            input_image=gray,
            render_factor=int(config.model["render_factor"]),
            postprocess_config=config.raw.get("postprocess", {}),
        )
        base.save(base_path)

        base_np = np.asarray(base)
        gray_np = np.asarray(gray)
        direct = Image.fromarray(runtime.apply(image_rgb=base_np, source_rgb=gray_np))
        direct.save(direct_path)

        hint_result = recolor_costume_with_hints(
            base_rgb=base_np,
            guide_gray_rgb=gray_np,
            runtime=runtime,
            max_side=int(args.max_side),
            seed_count=int(args.seed_count),
            iterations=int(args.iterations),
        )
        Image.fromarray(hint_result.recolored_rgb).save(hint_path)
        Image.fromarray(hint_result.seed_overlay_rgb).save(seeds_path)
        Image.fromarray(hint_result.mask_overlay_rgb).save(mask_path)
        Image.fromarray((hint_result.costume_mask * 255).astype(np.uint8)).save(costume_mask_path)

        items: list[tuple[str, Image.Image]] = [
            ("Original", original),
            ("B/W", gray),
            ("Base", base),
            ("Direct Palette", direct),
            ("Hint Recolor", Image.fromarray(hint_result.recolored_rgb)),
            ("Mask Overlay", Image.fromarray(hint_result.mask_overlay_rgb)),
            ("Seed Overlay", Image.fromarray(hint_result.seed_overlay_rgb)),
        ]
        sheet = build_item_sheet(items=items, font=font)
        sheet_path = sheet_dir / f"{safe}.png"
        sheet.save(sheet_path)
        sheets.append(sheet)
        summary["timestamps"].append(
            {
                "timestamp": timestamp,
                "sheet": str(sheet_path),
                "seed_count": len(hint_result.seeds),
                "base": str(base_path),
                "direct_palette": str(direct_path),
                "hint_recolor": str(hint_path),
                "mask_overlay": str(mask_path),
                "costume_mask": str(costume_mask_path),
                "seed_overlay": str(seeds_path),
                "change_metrics": {
                    "direct_palette": summarize_change_locality(
                        base_rgb=base_np,
                        variant_rgb=np.asarray(direct),
                        mask=hint_result.costume_mask.astype(bool),
                    ),
                    "hint_recolor": summarize_change_locality(
                        base_rgb=base_np,
                        variant_rgb=hint_result.recolored_rgb,
                        mask=hint_result.costume_mask.astype(bool),
                    ),
                },
            }
        )

    comparison_sheet_path = output_root / "comparison_sheet.png"
    stack_sheets(sheets).save(comparison_sheet_path)
    summary["comparison_sheet"] = str(comparison_sheet_path)
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(comparison_sheet_path)
    return 0


def build_runtime_from_args(*, config, args: argparse.Namespace, device: torch.device):
    postprocess = dict(config.raw.get("postprocess", {}))
    costume_palette = dict(postprocess.get("costume_palette", {}))
    costume_palette["enabled"] = True
    costume_palette["mask_backend"] = str(args.mask_backend)
    if args.reference_image:
        costume_palette["reference_images"] = list(args.reference_image)
    if args.palette_color:
        costume_palette["palette_colors"] = list(args.palette_color)
    postprocess["costume_palette"] = costume_palette
    runtime = build_costume_palette_runtime(postprocess_config=postprocess, device=device)
    if runtime is None:
        raise RuntimeError("Failed to build costume palette runtime.")
    return runtime


def build_item_sheet(*, items: list[tuple[str, Image.Image]], font: ImageFont.ImageFont) -> Image.Image:
    tile_size = (360, 240)
    margin = 20
    label_height = 30
    columns = 4
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            columns * tile_size[0] + (columns + 1) * margin,
            rows * (tile_size[1] + label_height) + (rows + 1) * margin,
        ),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        row = index // columns
        col = index % columns
        x = margin + col * (tile_size[0] + margin)
        y = margin + row * (tile_size[1] + label_height + margin)
        fitted = fit_image(image, tile_size)
        sheet.paste(fitted, (x, y))
        draw.text((x, y + tile_size[1] + 6), label, fill=(235, 235, 235), font=font)
    return sheet


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    src_w, src_h = image.size
    dst_w, dst_h = size
    scale = min(dst_w / src_w, dst_h / src_h)
    resized = image.resize(
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, color=(0, 0, 0))
    offset = ((dst_w - resized.width) // 2, (dst_h - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def stack_sheets(sheets: list[Image.Image]) -> Image.Image:
    if not sheets:
        raise ValueError("At least one sheet is required.")
    width = max(sheet.width for sheet in sheets)
    height = sum(sheet.height for sheet in sheets)
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    y = 0
    for sheet in sheets:
        canvas.paste(sheet, (0, y))
        y += sheet.height
    return canvas


def summarize_change_locality(*, base_rgb: np.ndarray, variant_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    delta = np.abs(variant_rgb.astype(np.float32) - base_rgb.astype(np.float32)).mean(axis=2)
    inside = float(delta[mask].mean()) if mask.any() else 0.0
    outside = float(delta[~mask].mean()) if (~mask).any() else 0.0
    return {
        "mean_rgb_abs_delta": float(delta.mean()),
        "inside_costume_mask_rgb_abs_delta": inside,
        "outside_costume_mask_rgb_abs_delta": outside,
        "inside_to_outside_ratio": float(inside / max(outside, 1e-6)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
