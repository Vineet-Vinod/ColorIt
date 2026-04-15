from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import cv2
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import load_config
from src.pipeline.ffmpeg_utils import encode_video_from_frames, extract_frames, ffprobe_media
from src.pipeline.refiner_runtime import load_refiner_bundle, refine_base_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a base DeOldify clip and a post-DeOldify refined clip.")
    parser.add_argument("--input-clip", required=True)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--config", default="configs/quality.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_clip = Path(args.input_clip).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()

    base_clip = output_root / "base.mp4"
    refined_clip = output_root / "refined.mp4"
    base_manifest = output_root / "render_manifest_base.json"
    frames_root = output_root / "frames"
    source_frames = frames_root / "source"
    base_frames = frames_root / "base"
    refined_frames = frames_root / "refined"
    if refined_frames.exists():
        shutil.rmtree(refined_frames)
    refined_frames.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    run_colorize_clip(
        config=config,
        config_path=config_path,
        input_path=input_clip,
        output_path=base_clip,
        manifest_path=base_manifest,
        overwrite=args.overwrite,
    )

    extract_frames(input_path=input_clip, output_dir=source_frames)
    extract_frames(input_path=base_clip, output_dir=base_frames)
    media = ffprobe_media(input_clip)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    bundle = load_refiner_bundle(checkpoint_path=Path(args.refiner_checkpoint), device=device)

    for source_frame in sorted(source_frames.glob("*.png")):
        base_frame = base_frames / source_frame.name
        refined_frame = refined_frames / source_frame.name
        gray = Image.open(source_frame).convert("RGB")
        base = Image.open(base_frame).convert("RGB")
        refined = refine_base_pil(bundle=bundle, base_image=base, gray_image=gray)
        refined.save(refined_frame)

    encode_video_from_frames(
        frame_dir=refined_frames,
        output_path=refined_clip,
        fps=str(media["fps"]),
        video_codec="libx264",
        crf=16,
        pixel_format="yuv420p",
        audio_input_path=input_clip,
    )
    print(refined_clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
