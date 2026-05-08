from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer


def run_model_chroma_propagate(
    *,
    source_path: Path,
    model_color_path: Path,
    output_path: Path,
    keyframe_stride: int,
    chroma_blend: float,
    overwrite: bool,
) -> int:
    source_path = source_path.expanduser().resolve()
    model_color_path = model_color_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source clip not found: {source_path}")
    if not model_color_path.exists():
        raise FileNotFoundError(f"Model color clip not found: {model_color_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    source_info = ffprobe_media(source_path)
    model_info = ffprobe_media(model_color_path)
    width = int(source_info["width"])
    height = int(source_info["height"])
    if width != int(model_info["width"]) or height != int(model_info["height"]):
        raise ValueError("Source and model color clips must have matching dimensions.")

    source_frames = _read_frames(source_path, width=width, height=height)
    model_frames = _read_frames(model_color_path, width=width, height=height)
    frame_count = min(len(source_frames), len(model_frames))
    source_frames = source_frames[:frame_count]
    model_frames = model_frames[:frame_count]
    if frame_count == 0:
        raise ValueError("No frames available for chroma propagation.")

    started = time.perf_counter()
    output_frames = _propagate_chroma(
        source_frames=source_frames,
        model_frames=model_frames,
        chroma_blend=chroma_blend,
    )
    _write_frames(
        output_path=output_path,
        frames=output_frames,
        width=width,
        height=height,
        fps=str(source_info["fps"]),
        audio_input_path=source_path,
    )
    print(f"Model chroma propagation written: {output_path}")
    print(f"Frames: {frame_count}")
    print("Chroma mode: direct")
    print(f"Runtime seconds: {time.perf_counter() - started:.2f}")
    return 0


def _propagate_chroma(
    *,
    source_frames: list[np.ndarray],
    model_frames: list[np.ndarray],
    chroma_blend: float,
) -> list[np.ndarray]:
    output_frames: list[np.ndarray] = []
    blend = float(np.clip(chroma_blend, 0.0, 1.0))
    for source_frame, model_frame in zip(source_frames, model_frames):
        source_lab = cv2.cvtColor(source_frame, cv2.COLOR_RGB2LAB)
        model_ab = cv2.cvtColor(model_frame, cv2.COLOR_RGB2LAB)[:, :, 1:3]
        if blend >= 1.0:
            source_lab[:, :, 1:3] = model_ab
        else:
            source_ab = source_lab[:, :, 1:3].astype(np.float32)
            output_ab = (1.0 - blend) * source_ab + blend * model_ab.astype(np.float32)
            source_lab[:, :, 1:3] = np.clip(output_ab, 0, 255).astype(np.uint8)
        output_frames.append(cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB))
    return output_frames


def _read_frames(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    frames: list[np.ndarray] = []
    try:
        while True:
            frame_data = reader.stdout.read(frame_bytes)
            if not frame_data:
                break
            if len(frame_data) != frame_bytes:
                raise RuntimeError(f"Unexpected rawvideo frame size: {len(frame_data)}")
            frames.append(np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3)).copy())
        returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if reader.stderr is not None:
        reader.stderr.close()
    return frames


def _write_frames(
    *,
    output_path: Path,
    frames: list[np.ndarray],
    width: int,
    height: int,
    fps: str,
    audio_input_path: Path,
) -> None:
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        video_codec="libx264",
        crf=16,
        pixel_format="yuv420p",
        audio_input_path=audio_input_path,
    )
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")
    try:
        for frame in frames:
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
        writer.stdin.close()
        returncode = writer.wait()
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {writer.stderr.read().decode().strip()}")
    writer.stderr.close()
