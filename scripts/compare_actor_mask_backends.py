from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

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
from src.pipeline.ffmpeg_utils import extract_single_frame
from src.pipeline.probes import parse_timecode_to_seconds


DEFAULT_FRAMES = [
    ("outdoor_three_actor", "00:58:28"),
    ("indoor_two_sari", "00:59:00"),
    ("close_pair", "01:00:40"),
    ("close_pair_side", "01:01:20"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare actor-mask backends on a small frame set.")
    parser.add_argument("--movie", default="../Movies/Kannada/emme thammanna.mp4")
    parser.add_argument(
        "--frame",
        action="append",
        default=[],
        help="Frame spec in the form frame_id=HH:MM:SS[.mmm]. Defaults to a built-in set.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help=f"Backend name. Defaults to: {', '.join(list_actor_mask_backends())}",
    )
    parser.add_argument("--output-root", default="data/experiments/actor_mask_backend_compare")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    movie_path = Path(args.movie).expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie not found: {movie_path}")

    frame_specs = parse_frame_specs(args.frame) if args.frame else DEFAULT_FRAMES
    backend_names = args.backend or list_actor_mask_backends()

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    frames_dir = output_root / "frames"
    masks_dir = output_root / "masks"
    overlays_dir = output_root / "overlays"
    sheets_dir = output_root / "sheets"
    for directory in (frames_dir, masks_dir, overlays_dir, sheets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    extracted_frames: dict[str, Path] = {}
    images: dict[str, Image.Image] = {}
    for frame_id, timecode in frame_specs:
        frame_path = frames_dir / f"{frame_id}.png"
        extract_single_frame(
            input_path=movie_path,
            output_path=frame_path,
            time_seconds=parse_timecode_to_seconds(timecode),
        )
        extracted_frames[frame_id] = frame_path
        images[frame_id] = Image.open(frame_path).convert("RGB")

    device = resolve_device(args.device)
    summary: dict[str, object] = {
        "movie_path": str(movie_path),
        "device": str(device),
        "frames": [],
        "backends": backend_names,
    }

    per_frame_results: dict[str, list[tuple[str, Image.Image]]] = {
        frame_id: [("original", images[frame_id].copy())] for frame_id, _ in frame_specs
    }

    for backend_name in backend_names:
        backend = load_actor_mask_backend(backend_name, device=device)
        for frame_id, timecode in frame_specs:
            image = images[frame_id]
            result = predict_actor_mask(backend, image)
            mask_path = masks_dir / frame_id / f"{backend_name}.png"
            overlay_path = overlays_dir / frame_id / f"{backend_name}.png"
            write_mask(mask_path, result.mask)
            write_actor_overlay(
                overlay_path,
                np.asarray(image),
                result.mask,
                label=f"{backend_name} {result.metadata['mask_fraction']:.3f}",
            )
            per_frame_results[frame_id].append((backend_name, Image.open(overlay_path).convert("RGB")))
            summary["frames"].append(
                {
                    "frame_id": frame_id,
                    "timecode": timecode,
                    "frame_path": str(extracted_frames[frame_id]),
                    "backend": backend_name,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "metadata": result.metadata,
                }
            )

    for frame_id, _ in frame_specs:
        build_frame_sheet(
            items=per_frame_results[frame_id],
            output_path=sheets_dir / f"{frame_id}_sheet.png",
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"output_root={output_root}")
    print(f"manifest={manifest_path}")
    return 0


def parse_frame_specs(values: list[str]) -> list[tuple[str, str]]:
    frame_specs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid frame spec: {value}")
        frame_id, timecode = value.split("=", 1)
        frame_specs.append((frame_id.strip(), timecode.strip()))
    return frame_specs


def build_frame_sheet(items: list[tuple[str, Image.Image]], output_path: Path) -> None:
    tile_size = (360, 240)
    margin = 20
    label_height = 30
    columns = 3
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
        draw.text((x, y + tile_size[1] + 6), label, fill=(235, 235, 235))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


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


if __name__ == "__main__":
    raise SystemExit(main())
