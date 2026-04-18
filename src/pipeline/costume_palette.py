from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from src.pipeline.actor_mask_backends import (
    ActorMaskBackend,
    load_actor_mask_backend,
    predict_actor_mask,
    resolve_device,
)
from src.pipeline.actor_masks import derive_costume_mask, derive_skin_mask


DEFAULT_MASK_BACKEND = "maskrcnn_v2_conservative"


@dataclass(frozen=True)
class PaletteColor:
    rgb: tuple[int, int, int]
    lightness: float
    ab: np.ndarray
    chroma: float


@dataclass
class CostumePaletteRuntime:
    actor_backend: ActorMaskBackend
    palette: list[PaletteColor]
    strength: float
    neutral_boost: float
    neutral_chroma: float
    min_chroma: float
    target_chroma_scale: float
    preserve_local_variation: float
    head_exclusion_fraction: float
    mask_blur_scale: float
    min_component_fraction: float
    max_components: int
    selection_luminance_weight: float
    selection_chroma_weight: float
    selection_chroma_reliability: float
    vividness_bias: float
    warm_bias: float

    def apply(
        self,
        *,
        image_rgb: np.ndarray,
        source_rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        return apply_costume_palette_guidance(
            image_rgb=image_rgb,
            runtime=self,
            source_rgb=source_rgb,
        )


def build_costume_palette_runtime(
    *,
    postprocess_config: dict,
    device: torch.device,
) -> CostumePaletteRuntime | None:
    config = postprocess_config.get("costume_palette", {})
    if not config or not bool(config.get("enabled", False)):
        return None

    palette: list[PaletteColor] = []
    palette.extend(_parse_palette_colors(config.get("palette_colors", [])))
    palette.extend(
        _extract_palette_from_references(
            reference_images=config.get("reference_images", []),
            palette_size=int(config.get("palette_size", 5)),
            min_reference_saturation=float(config.get("min_reference_saturation", 0.20)),
            max_reference_pixels=int(config.get("max_reference_pixels", 120000)),
        )
    )
    palette = _dedupe_palette(palette)
    if not palette:
        raise ValueError("costume_palette is enabled but no palette colors or usable reference images were provided.")

    backend_device_name = str(config.get("device", device.type if device.type in {"mps", "cpu"} else "cpu"))
    backend_device = resolve_device(backend_device_name)
    actor_backend = load_actor_mask_backend(str(config.get("mask_backend", DEFAULT_MASK_BACKEND)), device=backend_device)

    return CostumePaletteRuntime(
        actor_backend=actor_backend,
        palette=palette,
        strength=float(np.clip(config.get("strength", 0.72), 0.0, 1.0)),
        neutral_boost=float(np.clip(config.get("neutral_boost", 0.18), 0.0, 1.0)),
        neutral_chroma=float(max(0.0, config.get("neutral_chroma", 24.0))),
        min_chroma=float(max(0.0, config.get("min_chroma", 26.0))),
        target_chroma_scale=float(max(0.1, config.get("target_chroma_scale", 1.0))),
        preserve_local_variation=float(np.clip(config.get("preserve_local_variation", 0.35), 0.0, 1.0)),
        head_exclusion_fraction=float(np.clip(config.get("head_exclusion_fraction", 0.16), 0.0, 0.45)),
        mask_blur_scale=float(np.clip(config.get("mask_blur_scale", 0.012), 0.0, 0.10)),
        min_component_fraction=float(max(0.0, config.get("min_component_fraction", 0.003))),
        max_components=int(max(1, config.get("max_components", 6))),
        selection_luminance_weight=float(max(0.0, config.get("selection_luminance_weight", 1.0))),
        selection_chroma_weight=float(max(0.0, config.get("selection_chroma_weight", 0.30))),
        selection_chroma_reliability=float(max(1.0, config.get("selection_chroma_reliability", 36.0))),
        vividness_bias=float(max(0.0, config.get("vividness_bias", 0.05))),
        warm_bias=float(max(0.0, config.get("warm_bias", 0.10))),
    )


def apply_costume_palette_guidance(
    *,
    image_rgb: np.ndarray,
    runtime: CostumePaletteRuntime,
    source_rgb: np.ndarray | None = None,
) -> np.ndarray:
    image_rgb = np.ascontiguousarray(image_rgb).astype(np.uint8)
    mask_input = image_rgb if source_rgb is None else np.ascontiguousarray(source_rgb).astype(np.uint8)
    mask_result = predict_actor_mask(runtime.actor_backend, Image.fromarray(mask_input, mode="RGB"))
    person_mask = mask_result.mask.astype(np.uint8)
    if not person_mask.any():
        return image_rgb

    skin_mask = derive_skin_mask(image_rgb, person_mask)
    costume_mask = derive_costume_mask(person_mask, skin_mask)
    costume_mask = _suppress_head_region(costume_mask, person_mask, runtime.head_exclusion_fraction)
    labels, component_ids = _select_costume_components(
        costume_mask=costume_mask,
        min_component_fraction=runtime.min_component_fraction,
        max_components=runtime.max_components,
    )
    if not component_ids:
        return image_rgb

    feather = _build_feather_mask(costume_mask=costume_mask, blur_scale=runtime.mask_blur_scale)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    current_ab = lab[:, :, 1:3] - 128.0

    for component_id in component_ids:
        component_mask = labels == component_id
        mean_l = float(lightness[component_mask].mean())
        mean_ab = current_ab[component_mask].mean(axis=0)
        mean_chroma = float(np.linalg.norm(mean_ab))
        palette_color = _select_palette_color(runtime=runtime, mean_l=mean_l, mean_ab=mean_ab, mean_chroma=mean_chroma)
        pixel_ab = current_ab[component_mask]
        pixel_chroma = np.linalg.norm(pixel_ab, axis=1)
        neutral_boost = np.clip(
            (runtime.neutral_chroma - pixel_chroma) / max(runtime.neutral_chroma, 1e-6),
            0.0,
            1.0,
        ) * runtime.neutral_boost
        strength = np.clip(runtime.strength + neutral_boost, 0.0, 1.0) * feather[component_mask]
        target_ab = _build_target_ab(
            base_ab=pixel_ab,
            base_chroma=pixel_chroma,
            component_chroma=mean_chroma,
            palette_color=palette_color,
            runtime=runtime,
        )
        remapped_ab = ((1.0 - strength[:, None]) * pixel_ab) + (strength[:, None] * target_ab)
        remapped_ab = _enforce_chroma_floor(
            ab=remapped_ab,
            chroma_floor=min(runtime.min_chroma, palette_color.chroma * runtime.target_chroma_scale),
        )
        current_ab[component_mask] = remapped_ab

    remapped_lab = lab.copy()
    remapped_lab[:, :, 1:3] = np.clip(current_ab + 128.0, 0.0, 255.0)
    return cv2.cvtColor(remapped_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _build_target_ab(
    *,
    base_ab: np.ndarray,
    base_chroma: np.ndarray,
    component_chroma: float,
    palette_color: PaletteColor,
    runtime: CostumePaletteRuntime,
) -> np.ndarray:
    target_ab = np.repeat((palette_color.ab * runtime.target_chroma_scale)[None, :], base_ab.shape[0], axis=0)
    if runtime.preserve_local_variation <= 0.0 or component_chroma <= 1e-3:
        return target_ab

    local_ratio = np.clip(base_chroma / max(component_chroma, 1e-3), 0.65, 1.45)
    local_scale = (1.0 - runtime.preserve_local_variation) + (runtime.preserve_local_variation * local_ratio)
    return target_ab * local_scale[:, None]


def _enforce_chroma_floor(*, ab: np.ndarray, chroma_floor: float) -> np.ndarray:
    if chroma_floor <= 0.0 or ab.size == 0:
        return ab
    magnitude = np.linalg.norm(ab, axis=1)
    required_scale = np.maximum(1.0, chroma_floor / np.clip(magnitude, 1e-3, None))
    return ab * required_scale[:, None]


def _select_palette_color(
    *,
    runtime: CostumePaletteRuntime,
    mean_l: float,
    mean_ab: np.ndarray,
    mean_chroma: float,
) -> PaletteColor:
    reliability = float(np.clip(mean_chroma / runtime.selection_chroma_reliability, 0.0, 1.0))
    best = runtime.palette[0]
    best_score = float("inf")
    for palette_color in runtime.palette:
        luminance_score = runtime.selection_luminance_weight * abs(mean_l - palette_color.lightness)
        chroma_score = runtime.selection_chroma_weight * reliability * float(np.linalg.norm(mean_ab - palette_color.ab))
        vividness_bonus = runtime.vividness_bias * palette_color.chroma
        warm_score = max(float(palette_color.ab[0]), 0.0) + (0.35 * max(float(palette_color.ab[1]), 0.0))
        score = luminance_score + chroma_score - vividness_bonus - (runtime.warm_bias * warm_score)
        if score < best_score:
            best = palette_color
            best_score = score
    return best


def _select_costume_components(
    *,
    costume_mask: np.ndarray,
    min_component_fraction: float,
    max_components: int,
) -> tuple[np.ndarray, list[int]]:
    if not costume_mask.any():
        return np.zeros_like(costume_mask, dtype=np.int32), []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(costume_mask.astype(np.uint8), connectivity=8)
    min_pixels = max(1, int(round(costume_mask.shape[0] * costume_mask.shape[1] * min_component_fraction)))
    ranked = [
        (label, int(stats[label, cv2.CC_STAT_AREA]))
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return labels, [label for label, _ in ranked[:max_components]]


def _suppress_head_region(costume_mask: np.ndarray, person_mask: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0 or not person_mask.any():
        return costume_mask.astype(np.uint8)
    suppressed = costume_mask.copy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_mask.astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        top = int(stats[label, cv2.CC_STAT_TOP])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        cutoff = min(person_mask.shape[0], top + max(1, int(round(height * fraction))))
        suppressed[(labels == label) & (np.arange(person_mask.shape[0])[:, None] < cutoff)] = 0
    return suppressed


def _build_feather_mask(*, costume_mask: np.ndarray, blur_scale: float) -> np.ndarray:
    feather = costume_mask.astype(np.float32)
    if blur_scale <= 0.0:
        return feather
    kernel_size = max(3, int(round(min(costume_mask.shape[:2]) * blur_scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    feather = cv2.GaussianBlur(feather, (kernel_size, kernel_size), 0)
    max_value = float(feather.max())
    if max_value <= 1e-6:
        return costume_mask.astype(np.float32)
    return np.clip(feather / max_value, 0.0, 1.0)


def _parse_palette_colors(values: list[str]) -> list[PaletteColor]:
    colors: list[PaletteColor] = []
    for value in values:
        if not value:
            continue
        colors.append(_palette_color_from_rgb(_parse_color_string(str(value))))
    return colors


def _extract_palette_from_references(
    *,
    reference_images: list[str],
    palette_size: int,
    min_reference_saturation: float,
    max_reference_pixels: int,
) -> list[PaletteColor]:
    if not reference_images:
        return []

    pixels: list[np.ndarray] = []
    max_pixels_per_image = max(2000, max_reference_pixels // max(1, len(reference_images)))
    for reference_path in reference_images:
        path = Path(reference_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")
        image = Image.open(path).convert("RGB")
        pixels.append(
            _extract_reference_pixels(
                image=image,
                min_reference_saturation=min_reference_saturation,
                max_pixels=max_pixels_per_image,
            )
        )
    usable = [chunk for chunk in pixels if chunk.size > 0]
    if not usable:
        return []

    data = np.concatenate(usable, axis=0).astype(np.float32)
    cluster_count = min(max(1, palette_size), data.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
    _, labels, centers = cv2.kmeans(data, cluster_count, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.reshape(-1), minlength=cluster_count)

    palette: list[tuple[float, PaletteColor]] = []
    for count, center in zip(counts.tolist(), centers, strict=True):
        color = _palette_color_from_rgb(tuple(int(np.clip(round(channel), 0, 255)) for channel in center))
        score = float(count) + (color.chroma * 4.0)
        palette.append((score, color))
    palette.sort(key=lambda item: item[0], reverse=True)
    return [color for _, color in palette]


def _extract_reference_pixels(
    *,
    image: Image.Image,
    min_reference_saturation: float,
    max_pixels: int,
) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, 640.0 / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    mask = saturation >= float(np.clip(min_reference_saturation, 0.0, 1.0))
    pixels = rgb[mask]
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)
    if pixels.shape[0] > max_pixels:
        stride = max(1, pixels.shape[0] // max_pixels)
        pixels = pixels[::stride][:max_pixels]
    return pixels.astype(np.uint8)


def _dedupe_palette(palette: list[PaletteColor]) -> list[PaletteColor]:
    deduped: list[PaletteColor] = []
    for color in palette:
        if any(np.linalg.norm(color.ab - existing.ab) < 6.0 for existing in deduped):
            continue
        deduped.append(color)
    deduped.sort(key=lambda item: item.chroma, reverse=True)
    return deduped


def _palette_color_from_rgb(rgb: tuple[int, int, int]) -> PaletteColor:
    rgb_array = np.array([[list(rgb)]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    ab = lab[1:3] - 128.0
    return PaletteColor(
        rgb=rgb,
        lightness=float(lab[0]),
        ab=ab,
        chroma=float(np.linalg.norm(ab)),
    )


def _parse_color_string(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) == 3:
            hex_value = "".join(channel * 2 for channel in hex_value)
        if len(hex_value) != 6:
            raise ValueError(f"Invalid hex color: {value}")
        return tuple(int(hex_value[index : index + 2], 16) for index in (0, 2, 4))
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid RGB color: {value}")
        rgb = tuple(int(part) for part in parts)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"RGB color out of range: {value}")
        return rgb
    raise ValueError(f"Unsupported color format: {value}")
