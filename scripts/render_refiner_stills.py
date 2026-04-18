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

from src.pipeline.actor_mask_backends import (
    list_actor_mask_backends,
    load_actor_mask_backend,
    predict_actor_mask,
    resolve_device,
    write_actor_overlay,
    write_mask,
)
from src.pipeline.config import load_config
from src.pipeline.costume_palette import build_costume_palette_runtime
from src.pipeline.ffmpeg_utils import extract_single_frame
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.inference import colorize_pil_image
from src.pipeline.probes import parse_timecode_to_seconds
from src.pipeline.refiner_runtime import load_refiner_bundle, refine_base_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render still comparisons for base DeOldify versus the refiner, with optional actor-mask gating."
    )
    parser.add_argument("--movie", required=True)
    parser.add_argument("--timestamp", action="append", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mask-backend",
        action="append",
        default=[],
        choices=list_actor_mask_backends(),
        help="Optional actor-mask backend to test. May be provided multiple times.",
    )
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
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
    device = resolve_device(args.device)
    frame_dir = output_root / "frames"
    output_dir = output_root / "outputs"
    mask_dir = output_root / "masks"
    overlay_dir = output_root / "mask_overlays"
    sheets_dir = output_root / "sheets"
    for directory in (frame_dir, output_dir, mask_dir, overlay_dir, sheets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    base_bundle = load_colorizer_bundle(config)
    refiner_bundle = load_refiner_bundle(
        checkpoint_path=Path(args.refiner_checkpoint),
        device=device,
    )
    costume_palette_runtime = build_costume_palette_runtime(
        postprocess_config=config.raw.get("postprocess", {}),
        device=device,
    )
    font = ImageFont.load_default()

    summary = {
        "movie": str(movie),
        "config": str(config_path),
        "refiner_checkpoint": str(Path(args.refiner_checkpoint).expanduser().resolve()),
        "device": str(device),
        "refiner_input_channels": refiner_bundle.input_channels,
        "mask_mode": "input+output" if refiner_bundle.input_channels > 4 else "output-only",
        "costume_palette_enabled": costume_palette_runtime is not None,
        "mask_backends": args.mask_backend,
        "frames": [],
    }
    sheets: list[Image.Image] = []
    actor_backends: dict[str, object] = {}

    for timestamp in args.timestamp:
        safe = timestamp.replace(":", "_").replace(".", "_")
        timestamp_dir = output_dir / safe
        timestamp_dir.mkdir(parents=True, exist_ok=True)

        orig_path = frame_dir / f"{safe}_original.png"
        bw_path = timestamp_dir / "bw.png"
        base_path = timestamp_dir / "base.png"
        refined_path = timestamp_dir / "refined_unmasked.png"

        extract_single_frame(
            input_path=movie,
            output_path=orig_path,
            time_seconds=parse_timecode_to_seconds(timestamp),
        )
        original = Image.open(orig_path).convert("RGB")
        gray = original.convert("L").convert("RGB")
        gray.save(bw_path)
        base = colorize_pil_image(
            model_bundle=base_bundle,
            input_image=gray,
            render_factor=int(config.model["render_factor"]),
            postprocess_config=config.raw.get("postprocess", {}),
        )
        base.save(base_path)
        refined = refine_base_pil(bundle=refiner_bundle, base_image=base, gray_image=gray)
        refined.save(refined_path)

        items: list[tuple[str, Image.Image]] = [
            ("Original", original),
            ("B/W", gray),
            ("Base", base),
            ("Refined", refined),
        ]
        base_np = np.asarray(base, dtype=np.float32)
        refined_np = np.asarray(refined, dtype=np.float32)
        frame_record: dict[str, object] = {
            "timestamp": timestamp,
            "safe_id": safe,
            "original": str(orig_path),
            "bw": str(bw_path),
            "base": str(base_path),
            "refined_unmasked": str(refined_path),
            "masked_variants": [],
        }
        if costume_palette_runtime is not None:
            base_palette = Image.fromarray(
                costume_palette_runtime.apply(
                    image_rgb=np.asarray(base),
                    source_rgb=np.asarray(gray),
                )
            )
            refined_palette = Image.fromarray(
                costume_palette_runtime.apply(
                    image_rgb=np.asarray(refined),
                    source_rgb=np.asarray(gray),
                )
            )
            base_palette_path = timestamp_dir / "base_palette.png"
            refined_palette_path = timestamp_dir / "refined_palette.png"
            base_palette.save(base_palette_path)
            refined_palette.save(refined_palette_path)
            items.extend(
                [
                    ("Base+Palette", base_palette),
                    ("Refined+Palette", refined_palette),
                ]
            )
            frame_record["base_palette"] = str(base_palette_path)
            frame_record["refined_palette"] = str(refined_palette_path)

        for backend_name in args.mask_backend:
            if backend_name not in actor_backends:
                actor_backends[backend_name] = load_actor_mask_backend(backend_name, device=device)
            actor_backend = actor_backends[backend_name]
            mask_result = predict_actor_mask(actor_backend, gray)
            mask_path = mask_dir / safe / f"{backend_name}.png"
            overlay_path = overlay_dir / safe / f"{backend_name}.png"
            write_mask(mask_path, mask_result.mask)
            write_actor_overlay(
                overlay_path,
                np.asarray(gray),
                mask_result.mask,
                label=f"{backend_name} {mask_result.metadata['mask_fraction']:.3f}",
            )
            mask_image = Image.fromarray((mask_result.mask * 255).astype(np.uint8))
            masked_refined = refine_base_pil(
                bundle=refiner_bundle,
                base_image=base,
                gray_image=gray,
                mask_image=mask_image,
            )
            masked_path = timestamp_dir / f"refined_{backend_name}.png"
            masked_refined.save(masked_path)

            items.append((f"{backend_name} Mask", Image.open(overlay_path).convert("RGB")))
            items.append((f"{backend_name} Refined", masked_refined))

            masked_np = np.asarray(masked_refined, dtype=np.float32)
            frame_record["masked_variants"].append(
                {
                    "backend": backend_name,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "output_path": str(masked_path),
                    "mask_metadata": mask_result.metadata,
                    "change_metrics": summarize_change_locality(
                        base_rgb=base_np,
                        unmasked_rgb=refined_np,
                        masked_rgb=masked_np,
                        mask=mask_result.mask.astype(bool),
                    ),
                }
            )

        frame_sheet = build_item_sheet(items=items, font=font)
        frame_sheet_path = sheets_dir / f"{safe}.png"
        frame_sheet.save(frame_sheet_path)
        frame_record["sheet"] = str(frame_sheet_path)
        sheets.append(frame_sheet)
        summary["frames"].append(frame_record)

    sheet_path = output_root / "comparison_sheet.png"
    stack_sheets(sheets).save(sheet_path)
    summary["comparison_sheet"] = str(sheet_path)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(sheet_path)
    return 0


def summarize_change_locality(
    *,
    base_rgb: np.ndarray,
    unmasked_rgb: np.ndarray,
    masked_rgb: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    masked_delta = np.abs(masked_rgb - base_rgb).mean(axis=2)
    unmasked_delta = np.abs(unmasked_rgb - base_rgb).mean(axis=2)
    inside_mask = float(masked_delta[mask].mean()) if mask.any() else 0.0
    outside_mask = float(masked_delta[~mask].mean()) if (~mask).any() else 0.0
    unmasked_inside = float(unmasked_delta[mask].mean()) if mask.any() else 0.0
    unmasked_outside = float(unmasked_delta[~mask].mean()) if (~mask).any() else 0.0
    suppression = 1.0 - (outside_mask / max(unmasked_outside, 1e-6))
    return {
        "masked_mean_rgb_abs_delta": float(masked_delta.mean()),
        "masked_inside_mask_rgb_abs_delta": inside_mask,
        "masked_outside_mask_rgb_abs_delta": outside_mask,
        "unmasked_inside_mask_rgb_abs_delta": unmasked_inside,
        "unmasked_outside_mask_rgb_abs_delta": unmasked_outside,
        "outside_change_suppression_vs_unmasked": float(suppression),
        "inside_to_outside_ratio": float(inside_mask / max(outside_mask, 1e-6)),
    }


def build_item_sheet(*, items: list[tuple[str, Image.Image]], font: ImageFont.ImageFont) -> Image.Image:
    tile_size = (360, 240)
    margin = 20
    label_height = 30
    columns = 5
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
        column = index % columns
        x = margin + column * (tile_size[0] + margin)
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
    stacked = Image.new("RGB", (width, height), (8, 8, 8))
    y = 0
    for sheet in sheets:
        stacked.paste(sheet, (0, y))
        y += sheet.height
    return stacked


if __name__ == "__main__":
    raise SystemExit(main())
