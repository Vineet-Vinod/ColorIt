from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer


CHROMA_TEMPORAL_ALPHA = 0.25


def run_model_chroma_propagate(
    *,
    source_path: Path,
    model_color_path: Path,
    output_path: Path,
    keyframe_stride: int,
    chroma_blend: float,
    audio_input_path: Path | None = None,
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

    started = time.perf_counter()
    frame_count = _stream_propagate_chroma(
        source_path=source_path,
        model_color_path=model_color_path,
        output_path=output_path,
        width=width,
        height=height,
        fps=str(source_info["fps"]),
        chroma_blend=chroma_blend,
        audio_input_path=audio_input_path.expanduser().resolve() if audio_input_path is not None else source_path,
    )
    if frame_count == 0:
        raise ValueError("No frames available for chroma propagation.")
    print(f"Model chroma propagation written: {output_path}")
    print(f"Frames: {frame_count}")
    print("Chroma mode: temporal-direct-streaming")
    print(f"Runtime seconds: {time.perf_counter() - started:.2f}")
    return 0


def _stream_propagate_chroma(
    *,
    source_path: Path,
    model_color_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: str,
    chroma_blend: float,
    audio_input_path: Path,
) -> int:
    frame_bytes = width * height * 3
    blend = float(np.clip(chroma_blend, 0.0, 1.0))
    previous_ab: np.ndarray | None = None
    frame_count = 0

    source_reader = open_rawvideo_reader(input_path=source_path)
    model_reader = open_rawvideo_reader(input_path=model_color_path)
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
    _ensure_pipe(source_reader.stdout, source_reader.stderr, "source rawvideo reader")
    _ensure_pipe(model_reader.stdout, model_reader.stderr, "model rawvideo reader")
    _ensure_pipe(writer.stdin, writer.stderr, "rawvideo writer")

    try:
        while True:
            source_data = _read_exact_or_none(source_reader.stdout, frame_bytes)
            model_data = _read_exact_or_none(model_reader.stdout, frame_bytes)
            if source_data is None or model_data is None:
                break

            source_frame = np.frombuffer(source_data, dtype=np.uint8).reshape((height, width, 3))
            model_frame = np.frombuffer(model_data, dtype=np.uint8).reshape((height, width, 3))
            output_frame, previous_ab = _propagate_chroma_frame(
                source_frame=source_frame,
                model_frame=model_frame,
                previous_ab=previous_ab,
                chroma_blend=blend,
            )
            writer.stdin.write(np.ascontiguousarray(output_frame).tobytes())
            frame_count += 1

        writer.stdin.close()
        writer_returncode = writer.wait()
        source_returncode = source_reader.wait()
        model_returncode = model_reader.wait()
    finally:
        _close_pipe(source_reader.stdout)
        _close_pipe(model_reader.stdout)
        _close_pipe(writer.stdin)

    if source_returncode != 0:
        raise RuntimeError(f"ffmpeg source rawvideo reader failed: {_read_stderr(source_reader)}")
    if model_returncode != 0:
        raise RuntimeError(f"ffmpeg model rawvideo reader failed: {_read_stderr(model_reader)}")
    if writer_returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {_read_stderr(writer)}")
    _close_pipe(source_reader.stderr)
    _close_pipe(model_reader.stderr)
    _close_pipe(writer.stderr)
    return frame_count


def _propagate_chroma_frame(
    *,
    source_frame: np.ndarray,
    model_frame: np.ndarray,
    previous_ab: np.ndarray | None,
    chroma_blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_lab = cv2.cvtColor(source_frame, cv2.COLOR_RGB2LAB)
    model_ab = cv2.cvtColor(model_frame, cv2.COLOR_RGB2LAB)[:, :, 1:3].astype(np.float32)
    if previous_ab is None:
        smoothed_ab = model_ab
    else:
        smoothed_ab = CHROMA_TEMPORAL_ALPHA * model_ab + (1.0 - CHROMA_TEMPORAL_ALPHA) * previous_ab
    if chroma_blend >= 1.0:
        source_lab[:, :, 1:3] = np.clip(smoothed_ab, 0, 255).astype(np.uint8)
    else:
        source_ab = source_lab[:, :, 1:3].astype(np.float32)
        output_ab = (1.0 - chroma_blend) * source_ab + chroma_blend * smoothed_ab
        source_lab[:, :, 1:3] = np.clip(output_ab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB), smoothed_ab


def _read_exact_or_none(stream, size: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < size:
        chunk = stream.read(size - len(buffer))
        if not chunk:
            if not buffer:
                return None
            raise RuntimeError(f"Unexpected rawvideo frame size: {len(buffer)}")
        buffer.extend(chunk)
    return bytes(buffer)


def _ensure_pipe(pipe, stderr, name: str) -> None:
    if pipe is None or stderr is None:
        raise RuntimeError(f"ffmpeg {name} failed to expose required pipes.")


def _close_pipe(pipe) -> None:
    if pipe is not None and not pipe.closed:
        pipe.close()


def _read_stderr(process) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode().strip()
