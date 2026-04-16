from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def select_primary_person_mask(person_prob: np.ndarray, threshold: float) -> np.ndarray:
    binary = (person_prob >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        adaptive_threshold = max(float(person_prob.max()) * 0.55, 0.18)
        binary = (person_prob >= adaptive_threshold).astype(np.uint8)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return postprocess_binary_mask(binary)

    center_prior = build_center_prior(binary.shape[0], binary.shape[1])
    best_label = 1
    best_score = -1.0
    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        area = float(stats[label, cv2.CC_STAT_AREA])
        score = float((component * person_prob * center_prior).sum()) + area * 1e-4
        if score > best_score:
            best_score = score
            best_label = label
    primary = (labels == best_label).astype(np.uint8)
    return postprocess_binary_mask(primary)


def derive_skin_mask(image_rgb: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    if not person_mask.any():
        return np.zeros_like(person_mask, dtype=np.uint8)

    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:, :, 0]
    cr_channel = ycrcb[:, :, 1]
    cb_channel = ycrcb[:, :, 2]
    r_channel = image_rgb[:, :, 0]
    g_channel = image_rgb[:, :, 1]
    b_channel = image_rgb[:, :, 2]

    skin = (
        (cr_channel >= 128)
        & (cr_channel <= 185)
        & (cb_channel >= 85)
        & (cb_channel <= 140)
        & (y_channel >= 40)
        & (r_channel >= g_channel * 0.9)
        & (r_channel >= b_channel * 0.8)
    )

    upper_body_prior = np.linspace(1.0, 0.25, image_rgb.shape[0], dtype=np.float32)[:, None]
    center_prior = build_center_prior(image_rgb.shape[0], image_rgb.shape[1])
    weighted_skin = skin.astype(np.float32) * upper_body_prior * np.clip(center_prior * 1.4, 0.0, 1.0)
    skin_mask = (weighted_skin > 0.2).astype(np.uint8) * person_mask
    kernel = make_kernel(image_rgb.shape[0], image_rgb.shape[1], scale=0.01)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
    return skin_mask.astype(np.uint8)


def derive_costume_mask(person_mask: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    if not person_mask.any():
        return np.zeros_like(person_mask, dtype=np.uint8)
    kernel = make_kernel(person_mask.shape[0], person_mask.shape[1], scale=0.012)
    expanded_skin = cv2.dilate(skin_mask, kernel, iterations=1)
    costume_mask = person_mask.copy()
    costume_mask[expanded_skin > 0] = 0
    costume_mask = cv2.morphologyEx(costume_mask, cv2.MORPH_OPEN, kernel)
    costume_mask = cv2.morphologyEx(costume_mask, cv2.MORPH_CLOSE, kernel)
    return costume_mask.astype(np.uint8)


def postprocess_binary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = make_kernel(mask.shape[0], mask.shape[1], scale=0.012)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned.astype(np.uint8)


def make_kernel(height: int, width: int, scale: float) -> np.ndarray:
    size = max(3, int(round(min(height, width) * scale)))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_center_prior(height: int, width: int) -> np.ndarray:
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    return np.exp(-(xx**2 + yy**2) / (2.0 * 0.45**2))


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


def write_debug_overlay(
    path: Path,
    image_rgb: np.ndarray,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
    costume_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_rgb.astype(np.float32).copy()
    overlay[person_mask > 0] = overlay[person_mask > 0] * 0.65 + np.array([40, 120, 255], dtype=np.float32) * 0.35
    overlay[skin_mask > 0] = overlay[skin_mask > 0] * 0.5 + np.array([255, 140, 80], dtype=np.float32) * 0.5
    overlay[costume_mask > 0] = overlay[costume_mask > 0] * 0.5 + np.array([60, 220, 120], dtype=np.float32) * 0.5
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path)
