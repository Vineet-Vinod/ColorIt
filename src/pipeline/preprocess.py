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


def preprocess_rgb_batch(
    input_rgbs: list[np.ndarray],
    preprocessing_config: dict,
) -> list[np.ndarray]:
    histogram_equalization = preprocessing_config.get("histogram_equalization", {})
    if not bool(histogram_equalization.get("enabled", False)):
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
