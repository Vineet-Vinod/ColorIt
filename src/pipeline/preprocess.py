from __future__ import annotations

import cv2
import numpy as np


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
