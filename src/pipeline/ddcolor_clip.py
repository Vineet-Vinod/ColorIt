from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import torch

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer
from src.vendor.ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model


DEFAULT_DDCOLOR_WEIGHTS_PATH = Path("models/ddcolor/pytorch_model.bin")


def run_ddcolor_clip(
    *,
    input_path: Path,
    output_path: Path,
    weights_path: Path,
    input_size: int,
    device: str,
    output_preset: str,
    overwrite: bool,
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    weights_path = weights_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"DDColor weights not found: {weights_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    selected_device = _select_device(device)
    print(f"Input clip: {input_path}")
    print(f"Output clip: {output_path}")
    print("DDColor backend: vendored inference")
    print(f"Device: {selected_device}")
    print(f"Input size: {input_size}")

    model = build_ddcolor_model(
        DDColor,
        model_path=str(weights_path),
        input_size=input_size,
        model_size="large",
        device=selected_device,
    )
    colorizer = ColorizationPipeline(model, input_size=input_size, device=selected_device)

    media_info = ffprobe_media(input_path)
    width = int(media_info["width"])
    height = int(media_info["height"])
    fps = str(media_info["fps"])
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=input_path)
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        video_codec="libx264",
        crf=16,
        pixel_format="yuv420p",
        preset=output_preset,
        audio_input_path=input_path,
    )
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")

    frame_index = 0
    start = time.time()
    try:
        while True:
            frame_data = reader.stdout.read(frame_bytes)
            if not frame_data:
                break
            if len(frame_data) != frame_bytes:
                raise RuntimeError(
                    f"Unexpected end of rawvideo stream; expected {frame_bytes} bytes, got {len(frame_data)}."
                )
            frame_rgb = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            output_bgr = colorizer.process(frame_bgr)
            output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
            writer.stdin.write(np.ascontiguousarray(output_rgb).tobytes())
            frame_index += 1

        writer.stdin.close()
        writer_returncode = writer.wait()
        reader_returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()
        if writer.stdin is not None:
            writer.stdin.close()

    if reader_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if writer_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {writer.stderr.read().decode().strip()}")
    reader.stderr.close()
    writer.stderr.close()

    runtime = time.time() - start
    print(f"DDColor clip written: {output_path}")
    print(f"Frames: {frame_index}")
    print(f"Runtime seconds: {runtime:.2f}")
    return 0
def _select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)
