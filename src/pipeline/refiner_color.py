from __future__ import annotations

from dataclasses import dataclass

import cv2
import kornia.color as kcolor
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class LabDeltaStats:
    delta_ab_mean: float
    delta_ab_abs_mean: float


def rgb_to_lab_tensor(rgb: Tensor) -> Tensor:
    return kcolor.rgb_to_lab(rgb.clamp(0.0, 1.0))


def lab_to_rgb_tensor(lab: Tensor) -> Tensor:
    return kcolor.lab_to_rgb(lab)


def grayscale_rgb_from_rgb_tensor(rgb: Tensor) -> Tensor:
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return gray.repeat(1, 3, 1, 1)


def compose_refiner_input(
    *,
    base_rgb: Tensor,
    gray_rgb: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    parts = [base_rgb, gray_rgb[:, 0:1]]
    if mask is not None:
        parts.append(mask)
    return torch.cat(parts, dim=1)


def delta_ab_from_lab(*, base_lab: Tensor, target_lab: Tensor) -> Tensor:
    return target_lab[:, 1:3] - base_lab[:, 1:3]


def apply_delta_ab(*, base_lab: Tensor, delta_ab: Tensor) -> Tensor:
    refined = base_lab.clone()
    refined[:, 1:3] = refined[:, 1:3] + delta_ab
    refined[:, 0:1] = refined[:, 0:1].clamp(0.0, 100.0)
    refined[:, 1:2] = refined[:, 1:2].clamp(-127.0, 127.0)
    refined[:, 2:3] = refined[:, 2:3].clamp(-127.0, 127.0)
    return refined


def feather_mask(mask: Tensor, kernel_size: int = 9) -> Tensor:
    if kernel_size <= 1:
        return mask.clamp(0.0, 1.0)
    pad = kernel_size // 2
    return torch.nn.functional.avg_pool2d(mask.clamp(0.0, 1.0), kernel_size, stride=1, padding=pad)


def apply_refiner_delta(
    *,
    base_rgb: Tensor,
    predicted_delta_ab: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    base_lab = rgb_to_lab_tensor(base_rgb)
    delta_ab = predicted_delta_ab
    if mask is not None:
        delta_ab = delta_ab * feather_mask(mask)
    refined_lab = apply_delta_ab(base_lab=base_lab, delta_ab=delta_ab)
    return lab_to_rgb_tensor(refined_lab).clamp(0.0, 1.0)


def rgb_to_lab_np(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_rgb_np(image_lab: np.ndarray) -> np.ndarray:
    clipped = image_lab.copy()
    clipped[:, :, 0] = np.clip(clipped[:, :, 0], 0.0, 255.0)
    clipped[:, :, 1] = np.clip(clipped[:, :, 1], 0.0, 255.0)
    clipped[:, :, 2] = np.clip(clipped[:, :, 2], 0.0, 255.0)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2RGB)


def apply_delta_ab_np(
    *,
    base_rgb: np.ndarray,
    delta_ab: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    base_lab = rgb_to_lab_np(base_rgb)
    adjusted = delta_ab
    if mask is not None:
        adjusted = adjusted * np.clip(mask, 0.0, 1.0)
    base_lab[:, :, 1:3] += adjusted
    return lab_to_rgb_np(base_lab)


def summarize_delta(delta_ab: Tensor) -> LabDeltaStats:
    return LabDeltaStats(
        delta_ab_mean=float(delta_ab.mean().detach().cpu()),
        delta_ab_abs_mean=float(delta_ab.abs().mean().detach().cpu()),
    )
