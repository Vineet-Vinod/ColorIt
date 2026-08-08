from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path

from src.pipeline.ffmpeg_utils import (
    ffprobe_media,
    open_rawvideo_reader,
    open_rawvideo_writer,
    playable_cfr_frame_count,
)
from src.pipeline.progress import ProgressBar


def apply_luma_clahe(
    frame_rgb: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: int,
    strength: float,
) -> np.ndarray:
    """Apply contrast-limited histogram equalization to RGB frame luminance."""
    if strength <= 0.0:
        return frame_rgb

    tile_grid_size = max(1, int(tile_grid_size))
    strength = min(1.0, max(0.0, float(strength)))

    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clip_limit)),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    equalized_l = clahe.apply(l_channel)
    if strength < 1.0:
        equalized_l = cv2.addWeighted(l_channel, 1.0 - strength, equalized_l, strength, 0.0)
    equalized_lab = cv2.merge((equalized_l, a_channel, b_channel))
    return cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2RGB)


def equalize_clip_luma_clahe(
    *,
    input_path: Path,
    output_path: Path,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    strength: float = 0.70,
    crf: int = 16,
    include_audio: bool = True,
    progress_label: str = "CLAHE",
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input clip not found: {input_path}")

    media_info = ffprobe_media(input_path)
    width = int(media_info["width"])
    height = int(media_info["height"])
    frame_bytes = width * height * 3
    frame_count = 0

    reader = open_rawvideo_reader(input_path=input_path)
    writer = open_rawvideo_writer(
        output_path=output_path,
        width=width,
        height=height,
        fps=str(media_info["fps"]),
        video_codec="libx264",
        crf=crf,
        pixel_format="yuv420p",
        audio_input_path=input_path if include_audio else None,
    )
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")

    with ProgressBar(
        progress_label,
        total=playable_cfr_frame_count(media_info),
        unit="frame",
    ) as progress:
        try:
            while True:
                frame_data = reader.stdout.read(frame_bytes)
                if not frame_data:
                    break
                if len(frame_data) != frame_bytes:
                    raise RuntimeError(
                        f"Unexpected end of rawvideo stream; expected {frame_bytes} bytes, got {len(frame_data)}."
                    )
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                equalized = apply_luma_clahe(
                    frame,
                    clip_limit=clip_limit,
                    tile_grid_size=tile_grid_size,
                    strength=strength,
                )
                writer.stdin.write(np.ascontiguousarray(equalized).tobytes())
                frame_count += 1
                progress.update()

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
    return frame_count


def apply_luma_clahe_to_colored_frame(
    *,
    source_rgb: np.ndarray,
    colored_rgb: np.ndarray,
    clip_limit: float,
    tile_grid_size: int,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return colored_rgb

    equalized_source = apply_luma_clahe(
        source_rgb,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        strength=strength,
    )
    colored_yuv = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2YUV)
    colored_y = colored_yuv[:, :, 0]
    equalized_y = cv2.cvtColor(equalized_source, cv2.COLOR_RGB2YUV)[:, :, 0]
    highlight_mask = (colored_y.astype(np.float32) / 255.0) > 0.74
    blended_y = np.where(highlight_mask, colored_y, equalized_y)
    colored_yuv[:, :, 0] = blended_y.astype(np.uint8)
    return cv2.cvtColor(colored_yuv, cv2.COLOR_YUV2RGB)


def apply_luma_clahe_to_colored_lab(
    colored_rgb: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: int,
    strength: float,
) -> np.ndarray:
    if strength <= 0.0:
        return colored_rgb

    tile_grid_size = max(1, int(tile_grid_size))
    strength = min(1.0, max(0.0, float(strength)))

    lab = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(clip_limit)),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    equalized_l = clahe.apply(l_channel)
    if strength < 1.0:
        equalized_l = cv2.addWeighted(l_channel, 1.0 - strength, equalized_l, strength, 0.0)
    return cv2.cvtColor(cv2.merge((equalized_l, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def preprocess_rgb_batch(
    input_rgbs: list[np.ndarray],
    preprocessing_config: dict,
) -> list[np.ndarray]:
    histogram_equalization = preprocessing_config.get("histogram_equalization", {})
    if not bool(histogram_equalization.get("enabled", False)):
        return input_rgbs
    if str(histogram_equalization.get("target", "model_input")) != "model_input":
        return input_rgbs

    clip_limit = float(histogram_equalization.get("clip_limit", 2.0))
    tile_grid_size = int(histogram_equalization.get("tile_grid_size", 8))
    strength = float(histogram_equalization.get("strength", 0.75))
    return [
        apply_luma_clahe(
            np.ascontiguousarray(frame),
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            strength=strength,
        )
        for frame in input_rgbs
    ]


def postprocess_colored_batch(
    *,
    source_rgbs: list[np.ndarray],
    colored_rgbs: list[np.ndarray],
    preprocessing_config: dict,
) -> list[np.ndarray]:
    histogram_equalization = preprocessing_config.get("histogram_equalization", {})
    if not bool(histogram_equalization.get("enabled", False)):
        return colored_rgbs
    target = str(histogram_equalization.get("target", "model_input"))
    if target not in {"output_luma", "colored_lab_luma"}:
        return colored_rgbs

    clip_limit = float(histogram_equalization.get("clip_limit", 2.0))
    tile_grid_size = int(histogram_equalization.get("tile_grid_size", 8))
    strength = float(histogram_equalization.get("strength", 0.75))
    if target == "colored_lab_luma":
        return [
            apply_luma_clahe_to_colored_lab(
                np.ascontiguousarray(colored_frame),
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                strength=strength,
            )
            for colored_frame in colored_rgbs
        ]

    return [
        apply_luma_clahe_to_colored_frame(
            source_rgb=np.ascontiguousarray(source_frame),
            colored_rgb=np.ascontiguousarray(colored_frame),
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            strength=strength,
        )
        for source_frame, colored_frame in zip(source_rgbs, colored_rgbs, strict=True)
    ]
