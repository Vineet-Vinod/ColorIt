"""Keep labeled materials on their assigned Lab chroma without changing source luminance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class MaterialLockSettings:
    close_radius: int = 7
    expand_radius: int = 5
    protect_radius: int = 5
    feather_radius: int = 7

    def __post_init__(self) -> None:
        for name in ("close_radius", "expand_radius", "protect_radius", "feather_radius"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


def lock_material_chroma(
    *,
    source_bgr: np.ndarray,
    color_bgr: np.ndarray,
    semantic_labels: Iterable[np.ndarray],
    material_label: int,
    material_color: str,
    protected_labels: Iterable[int] = (),
    settings: MaterialLockSettings = MaterialLockSettings(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return a source-luma frame and the feathered material correction mask.

    Multiple semantic maps may be supplied, for example one inferred from the
    source and another from the generated color frame. Their material and
    protected regions are combined before morphology.
    """
    _validate_frame("source_bgr", source_bgr)
    _validate_frame("color_bgr", color_bgr)
    if source_bgr.shape != color_bgr.shape:
        raise ValueError("source_bgr and color_bgr must have the same shape")

    label_maps = tuple(np.asarray(labels) for labels in semantic_labels)
    if not label_maps:
        raise ValueError("At least one semantic label map is required")
    height, width = source_bgr.shape[:2]
    for labels in label_maps:
        if labels.shape != (height, width):
            raise ValueError("Semantic label maps must match the frame dimensions")

    material = np.zeros((height, width), dtype=np.uint8)
    protected = np.zeros((height, width), dtype=np.uint8)
    protected_ids = tuple(int(value) for value in protected_labels)
    for labels in label_maps:
        material |= (labels == int(material_label)).astype(np.uint8)
        for label in protected_ids:
            protected |= (labels == label).astype(np.uint8)

    material = _morph(material, cv2.MORPH_CLOSE, settings.close_radius)
    material = _dilate(material, settings.expand_radius)
    protected = _dilate(protected, settings.protect_radius)
    material[protected > 0] = 0
    alpha = _feather(material, settings.feather_radius)

    source_luma = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_chroma = color_lab[:, :, 1:3] - 128.0
    target_chroma = _hex_to_lab_chroma(material_color)
    locked_chroma = (
        current_chroma * (1.0 - alpha[:, :, None])
        + target_chroma[None, None, :] * alpha[:, :, None]
    )
    color_lab[:, :, 0] = source_luma
    color_lab[:, :, 1:3] = np.clip(locked_chroma + 128.0, 0.0, 255.0)
    output = cv2.cvtColor(color_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return output, alpha


def _validate_frame(name: str, frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 uint8 array")


def _odd_kernel(radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    size = radius if radius % 2 == 1 else radius + 1
    return np.ones((size, size), dtype=np.uint8)


def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
    kernel = _odd_kernel(radius)
    if kernel is None:
        return mask
    return cv2.morphologyEx(mask, operation, kernel)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = _odd_kernel(radius)
    if kernel is None:
        return mask
    return cv2.dilate(mask, kernel, iterations=1)


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    size = radius if radius % 2 == 1 else radius + 1
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (size, size), 0), 0.0, 1.0)


def _hex_to_lab_chroma(value: str) -> np.ndarray:
    text = value.removeprefix("#")
    if len(text) != 6 or any(character not in "0123456789abcdefABCDEF" for character in text):
        raise ValueError(f"Expected #RRGGBB material color, got {value!r}")
    red, green, blue = (int(text[index : index + 2], 16) for index in (0, 2, 4))
    pixel = np.array([[[blue, green, red]]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, 1:3] - 128.0
