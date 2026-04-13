from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a review clip and comparison sheets for each epoch checkpoint.")
    parser.add_argument("--input-clip", required=True, help="Source clip to colorize for comparison.")
    parser.add_argument("--base-checkpoint", default="models/deoldify/ColorizeVideo_gen.pth")
    parser.add_argument("--epoch-dir", required=True, help="Directory containing epoch_XX.pth checkpoints.")
    parser.add_argument("--output-root", required=True, help="Directory for rendered clips and sheets.")
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--frame-index", action="append", type=int, default=[], help="Frame index to extract. Repeat as needed.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_clip = Path(args.input_clip).expanduser().resolve()
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    epoch_dir = Path(args.epoch_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    frame_indices = args.frame_index or [0, 240, 479]

    if not input_clip.exists():
        raise FileNotFoundError(f"Input clip not found: {input_clip}")
    if not epoch_dir.exists():
        raise FileNotFoundError(f"Epoch directory not found: {epoch_dir}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    clips_dir = output_root / "clips"
    frames_dir = output_root / "frames"
    sheets_dir = output_root / "sheets"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    source_frame_dir = frames_dir / "source"
    extract_frames(input_clip, source_frame_dir, frame_indices)

    base_output = clips_dir / "base.mp4"
    render_clip(
        input_clip=input_clip,
        config_path=config_path,
        checkpoint_path=base_checkpoint,
        output_path=base_output,
        manifest_path=output_root / "render_manifest_base.json",
    )
    base_frame_dir = frames_dir / "base"
    extract_frames(base_output, base_frame_dir, frame_indices)

    epoch_checkpoints = sorted(epoch_dir.glob("epoch_*.pth"))
    if not epoch_checkpoints:
        raise RuntimeError(f"No epoch checkpoints found in {epoch_dir}")

    for checkpoint_path in epoch_checkpoints:
        epoch_name = checkpoint_path.stem
        clip_output = clips_dir / f"{epoch_name}.mp4"
        render_clip(
            input_clip=input_clip,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_path=clip_output,
            manifest_path=output_root / f"render_manifest_{epoch_name}.json",
        )
        epoch_frame_dir = frames_dir / epoch_name
        extract_frames(clip_output, epoch_frame_dir, frame_indices)
        build_sheet(
            output_path=sheets_dir / f"{epoch_name}.png",
            source_dir=source_frame_dir,
            base_dir=base_frame_dir,
            epoch_dir=epoch_frame_dir,
            frame_indices=frame_indices,
            epoch_name=epoch_name,
        )

    print(f"Comparison outputs written to {output_root}")
    return 0


def render_clip(
    *,
    input_clip: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> None:
    config = load_config(config_path)
    config.raw["model"]["weights_path"] = str(checkpoint_path)
    config.raw["runtime"]["inference_batch_size"] = 16
    config.raw["runtime"]["num_workers"] = 0
    config.raw["runtime"]["cleanup_frames"] = True
    config.raw["postprocess"]["temporal_smoothing"] = False
    run_colorize_clip(
        config=config,
        config_path=config_path,
        input_path=input_clip,
        output_path=output_path,
        manifest_path=manifest_path,
        overwrite=True,
    )


def extract_frames(video_path: Path, output_dir: Path, frame_indices: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='{selection}'",
        "-vsync",
        "0",
        str(output_dir / "frame_%02d.png"),
    ]
    subprocess.run(command, check=True)


def build_sheet(
    *,
    output_path: Path,
    source_dir: Path,
    base_dir: Path,
    epoch_dir: Path,
    frame_indices: list[int],
    epoch_name: str,
) -> None:
    labels = [("Source", source_dir), ("Base", base_dir), (epoch_name, epoch_dir)]
    rows: list[Image.Image] = []
    header_height = 36
    for row_index, _frame_index in enumerate(frame_indices, start=1):
        images = []
        for label, directory in labels:
            image = Image.open(directory / f"frame_{row_index:02d}.png").convert("RGB")
            images.append((label, image))
        width, height = images[0][1].size
        row = Image.new("RGB", (width * len(images), height + header_height), "white")
        draw = ImageDraw.Draw(row)
        for column_index, (label, image) in enumerate(images):
            x_offset = column_index * width
            row.paste(image, (x_offset, header_height))
            draw.text((x_offset + 12, 10), label, fill="black")
        rows.append(row)

    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y_offset = 0
    for row in rows:
        canvas.paste(row, (0, y_offset))
        y_offset += row.height
    canvas.save(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
