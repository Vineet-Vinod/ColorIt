from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import (
    ffprobe_media,
    open_rawvideo_reader,
    open_rawvideo_writer,
    playable_cfr_frame_count,
)
from src.pipeline.progress import ProgressBar


DEFAULT_BACKGROUND_CHROMA_HOLD = {
    "enabled": True,
    "blend": 0.72,
    "update_alpha": 0.035,
    "luma_tolerance": 10.0,
    "previous_luma_tolerance": 8.0,
    "shot_change_threshold": 18.0,
    "min_luma": 25.0,
    "max_luma": 245.0,
    "skin_min_a": 130.0,
    "skin_max_a": 170.0,
    "skin_min_b": 130.0,
    "skin_max_b": 178.0,
    "skin_min_chroma": 6.0,
    "feather_radius": 11,
}


def run_background_chroma_hold(
    *,
    input_path: Path,
    output_path: Path,
    settings: dict[str, object] | None = None,
    overwrite: bool,
    progress_label: str = "Background chroma hold",
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    merged_settings = dict(DEFAULT_BACKGROUND_CHROMA_HOLD)
    if settings is not None:
        merged_settings.update(settings)

    media_info = ffprobe_media(input_path)
    started = time.perf_counter()
    frame_count = _stream_hold_chroma(
        input_path=input_path,
        output_path=output_path,
        media_info=media_info,
        settings=merged_settings,
        progress_label=progress_label,
    )
    print(f"Background chroma hold written: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Runtime seconds: {time.perf_counter() - started:.2f}")
    return 0


def _stream_hold_chroma(
    *,
    input_path: Path,
    output_path: Path,
    media_info: dict[str, str | int | float],
    settings: dict[str, object],
    progress_label: str,
) -> int:
    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=input_path)
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=str(media_info["fps"]),
        video_codec="libx264",
        crf=16,
        pixel_format="yuv420p",
        audio_input_path=input_path,
    )
    _ensure_pipe(reader.stdout, reader.stderr, "rawvideo reader")
    _ensure_pipe(writer.stdin, writer.stderr, "rawvideo writer")

    reference_luma: np.ndarray | None = None
    held_ab: np.ndarray | None = None
    previous_luma: np.ndarray | None = None
    frame_count = 0
    with ProgressBar(
        progress_label,
        total=playable_cfr_frame_count(media_info),
        unit="frame",
    ) as progress:
        try:
            while True:
                frame_data = _read_exact_or_none(reader.stdout, frame_bytes)
                if frame_data is None:
                    break

                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
                luma = lab[:, :, 0]
                ab = lab[:, :, 1:3]
                if reference_luma is None or held_ab is None or previous_luma is None:
                    reference_luma = luma.copy()
                    held_ab = ab.copy()
                    previous_luma = luma.copy()
                    output_frame = frame
                else:
                    frame_delta = float(np.mean(np.abs(luma - previous_luma)))
                    if frame_delta >= float(settings["shot_change_threshold"]):
                        reference_luma = luma.copy()
                        held_ab = ab.copy()
                        output_frame = frame
                    else:
                        output_lab, reference_luma, held_ab = _hold_frame_chroma(
                            lab=lab,
                            reference_luma=reference_luma,
                            held_ab=held_ab,
                            previous_luma=previous_luma,
                            settings=settings,
                        )
                        output_frame = cv2.cvtColor(
                            np.clip(output_lab, 0, 255).astype(np.uint8),
                            cv2.COLOR_LAB2RGB,
                        )
                    previous_luma = luma.copy()

                writer.stdin.write(np.ascontiguousarray(output_frame).tobytes())
                frame_count += 1
                progress.update()

            writer.stdin.close()
            writer_returncode = writer.wait()
            reader_returncode = reader.wait()
        finally:
            _close_pipe(reader.stdout)
            _close_pipe(writer.stdin)

        if reader_returncode != 0:
            raise RuntimeError(f"ffmpeg rawvideo reader failed: {_read_stderr(reader)}")
        if writer_returncode != 0:
            raise RuntimeError(f"ffmpeg rawvideo writer failed: {_read_stderr(writer)}")
        _close_pipe(reader.stderr)
        _close_pipe(writer.stderr)
    return frame_count


def _hold_frame_chroma(
    *,
    lab: np.ndarray,
    reference_luma: np.ndarray,
    held_ab: np.ndarray,
    previous_luma: np.ndarray,
    settings: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    luma = lab[:, :, 0]
    ab = lab[:, :, 1:3]
    stable_mask = (
        (luma >= float(settings["min_luma"]))
        & (luma <= float(settings["max_luma"]))
        & (np.abs(luma - reference_luma) <= float(settings["luma_tolerance"]))
        & (np.abs(luma - previous_luma) <= float(settings["previous_luma_tolerance"]))
        & ~_skin_like_mask(lab=lab, settings=settings)
    )
    if not np.any(stable_mask):
        return lab, reference_luma, held_ab

    update_alpha = float(np.clip(settings["update_alpha"], 0.0, 1.0))
    held_ab[stable_mask] = (1.0 - update_alpha) * held_ab[stable_mask] + update_alpha * ab[stable_mask]
    reference_luma[stable_mask] = 0.98 * reference_luma[stable_mask] + 0.02 * luma[stable_mask]

    feather = _feather_mask(stable_mask, radius=int(settings["feather_radius"]))
    blend = float(np.clip(settings["blend"], 0.0, 1.0))
    alpha = feather * blend
    output_lab = lab.copy()
    output_lab[:, :, 1:3] = ab * (1.0 - alpha[:, :, None]) + held_ab * alpha[:, :, None]
    return output_lab, reference_luma, held_ab


def _skin_like_mask(*, lab: np.ndarray, settings: dict[str, object]) -> np.ndarray:
    luma = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    chroma = np.sqrt((a - 128.0) * (a - 128.0) + (b - 128.0) * (b - 128.0))
    return (
        (luma >= 45.0)
        & (luma <= 235.0)
        & (a >= float(settings["skin_min_a"]))
        & (a <= float(settings["skin_max_a"]))
        & (b >= float(settings["skin_min_b"]))
        & (b <= float(settings["skin_max_b"]))
        & (chroma >= float(settings["skin_min_chroma"]))
    )


def _feather_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    kernel_size = radius if radius % 2 == 1 else radius + 1
    feather = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
    return np.clip(feather, 0.0, 1.0)


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
