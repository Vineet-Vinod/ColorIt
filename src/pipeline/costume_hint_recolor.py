from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pipeline.actor_masks import derive_costume_mask, derive_skin_mask
from src.pipeline.costume_palette import CostumePaletteRuntime


@dataclass(frozen=True)
class CostumeSeed:
    x: int
    y: int
    rgb: tuple[int, int, int]


@dataclass(frozen=True)
class CostumeHintRecolorResult:
    recolored_rgb: np.ndarray
    person_mask: np.ndarray
    skin_mask: np.ndarray
    costume_mask: np.ndarray
    seed_overlay_rgb: np.ndarray
    mask_overlay_rgb: np.ndarray
    seeds: list[CostumeSeed]


def recolor_costume_with_hints(
    *,
    base_rgb: np.ndarray,
    guide_gray_rgb: np.ndarray,
    runtime: CostumePaletteRuntime,
    max_side: int = 384,
    seed_count: int = 8,
    iterations: int = 160,
    edge_beta: float = 16.0,
    anchor_strength: float = 0.02,
    component_seed_share: float = 0.75,
) -> CostumeHintRecolorResult:
    base_rgb = np.ascontiguousarray(base_rgb).astype(np.uint8)
    guide_gray_rgb = np.ascontiguousarray(guide_gray_rgb).astype(np.uint8)

    from src.pipeline.actor_mask_backends import predict_actor_mask

    mask_result = predict_actor_mask(runtime.actor_backend, _rgb_to_pil(guide_gray_rgb))
    person_mask = mask_result.mask.astype(np.uint8)
    skin_mask = derive_skin_mask(base_rgb, person_mask)
    costume_mask = derive_costume_mask(person_mask, skin_mask)
    costume_mask = _suppress_head_region(costume_mask, person_mask, skin_mask, runtime.head_exclusion_fraction)
    costume_mask = _suppress_hair_like_regions(base_rgb, costume_mask, person_mask, skin_mask)

    if not costume_mask.any():
        return CostumeHintRecolorResult(
            recolored_rgb=base_rgb,
            person_mask=person_mask,
            skin_mask=skin_mask,
            costume_mask=costume_mask,
            seed_overlay_rgb=base_rgb.copy(),
            mask_overlay_rgb=_build_mask_overlay(base_rgb, person_mask, skin_mask, costume_mask),
            seeds=[],
        )

    scale = min(1.0, float(max_side) / float(max(base_rgb.shape[:2])))
    small_rgb = _resize_rgb(base_rgb, scale)
    small_gray = _resize_rgb(guide_gray_rgb, scale)
    small_person = _resize_mask(person_mask, scale)
    small_skin = _resize_mask(skin_mask, scale)
    small_costume = _resize_mask(costume_mask, scale)

    small_lab = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    small_luma = cv2.cvtColor(small_gray, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    base_ab = small_lab[:, :, 1:3] - 128.0

    labels, component_ids = _select_components(
        small_costume,
        min_component_fraction=runtime.min_component_fraction,
        max_components=runtime.max_components,
    )
    if not component_ids:
        return CostumeHintRecolorResult(
            recolored_rgb=base_rgb,
            person_mask=person_mask,
            skin_mask=skin_mask,
            costume_mask=costume_mask,
            seed_overlay_rgb=base_rgb.copy(),
            mask_overlay_rgb=_build_mask_overlay(base_rgb, person_mask, skin_mask, costume_mask),
            seeds=[],
        )

    seed_delta = np.zeros((small_rgb.shape[0], small_rgb.shape[1], 2), dtype=np.float32)
    seed_mask = np.zeros((small_rgb.shape[0], small_rgb.shape[1]), dtype=bool)
    all_small_seeds: list[tuple[int, int, tuple[int, int, int]]] = []

    refined_components = _refine_component_masks(
        image_rgb=small_rgb,
        person_mask=small_person,
        skin_mask=small_skin,
        labels=labels,
        component_ids=component_ids,
    )
    if not refined_components:
        return CostumeHintRecolorResult(
            recolored_rgb=base_rgb,
            person_mask=person_mask,
            skin_mask=skin_mask,
            costume_mask=costume_mask,
            seed_overlay_rgb=base_rgb.copy(),
            mask_overlay_rgb=_build_mask_overlay(base_rgb, person_mask, skin_mask, costume_mask),
            seeds=[],
        )

    palette_assignment = _assign_palette_colors(
        runtime=runtime,
        component_masks=refined_components,
        small_lab=small_lab,
    )

    component_budget = max(1, int(round(seed_count * float(component_seed_share))))
    propagated_delta = np.zeros_like(seed_delta)
    refined_union = np.zeros_like(small_costume, dtype=np.uint8)
    for component_index, component_mask in enumerate(refined_components):
        if not component_mask.any():
            continue
        refined_union = np.maximum(refined_union, component_mask.astype(np.uint8))
        palette_color = palette_assignment[component_index]
        anchor_mask = _build_anchor_mask(component_mask.astype(np.uint8))
        if not anchor_mask.any():
            anchor_mask = component_mask.astype(np.uint8)
        target_ab = palette_color.ab * runtime.target_chroma_scale
        target_field = _build_target_field(
            component_mask=component_mask,
            anchor_mask=anchor_mask.astype(bool),
            base_ab=base_ab,
            target_ab=target_ab,
            preserve_local_variation=runtime.preserve_local_variation,
            min_chroma=runtime.min_chroma,
        )
        component_seed_mask = anchor_mask.astype(bool)
        component_seed_delta = np.zeros_like(seed_delta)
        component_seed_delta[component_seed_mask] = target_field[component_seed_mask] - base_ab[component_seed_mask]
        component_delta = _propagate_seed_deltas(
            luma=small_luma,
            mask=component_mask.astype(bool),
            seed_mask=component_seed_mask,
            seed_delta=component_seed_delta,
            iterations=max(iterations, 220),
            edge_beta=edge_beta * 0.65,
            anchor_strength=anchor_strength,
        )
        propagated_delta += component_delta
        seed_mask |= component_seed_mask
        seed_delta[component_seed_mask] = component_seed_delta[component_seed_mask]

        component_seeds = _pick_seed_points(component_mask.astype(np.uint8), max(1, component_budget // len(refined_components)))
        for seed_y, seed_x in component_seeds:
            all_small_seeds.append((seed_y, seed_x, palette_color.rgb))

    if not seed_mask.any():
        return CostumeHintRecolorResult(
            recolored_rgb=base_rgb,
            person_mask=person_mask,
            skin_mask=skin_mask,
            costume_mask=costume_mask,
            seed_overlay_rgb=base_rgb.copy(),
            mask_overlay_rgb=_build_mask_overlay(base_rgb, person_mask, skin_mask, costume_mask),
            seeds=[],
        )

    feather = _build_feather_mask(costume_mask=refined_union, blur_scale=runtime.mask_blur_scale * 0.5)
    if scale < 1.0:
        full_delta = cv2.resize(
            propagated_delta,
            (base_rgb.shape[1], base_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        full_feather = cv2.resize(
            feather,
            (base_rgb.shape[1], base_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        full_delta = propagated_delta
        full_feather = feather
    recolored_rgb = _apply_delta_to_base(
        base_rgb=base_rgb,
        delta_ab=full_delta,
        feather=np.clip(full_feather, 0.0, 1.0),
        strength=runtime.strength,
        min_chroma=runtime.min_chroma,
    )

    full_seeds = [
        CostumeSeed(
            x=int(round(seed_x / max(scale, 1e-6))),
            y=int(round(seed_y / max(scale, 1e-6))),
            rgb=rgb,
        )
        for seed_y, seed_x, rgb in all_small_seeds
    ]

    return CostumeHintRecolorResult(
        recolored_rgb=recolored_rgb,
        person_mask=person_mask,
        skin_mask=skin_mask,
        costume_mask=_resize_mask(refined_union, 1.0 / scale) if scale < 1.0 else refined_union,
        seed_overlay_rgb=_draw_seed_overlay(base_rgb, full_seeds),
        mask_overlay_rgb=_build_mask_overlay(
            base_rgb,
            person_mask,
            skin_mask,
            _resize_mask(refined_union, 1.0 / scale) if scale < 1.0 else refined_union,
        ),
        seeds=full_seeds,
    )


def _propagate_seed_deltas(
    *,
    luma: np.ndarray,
    mask: np.ndarray,
    seed_mask: np.ndarray,
    seed_delta: np.ndarray,
    iterations: int,
    edge_beta: float,
    anchor_strength: float,
) -> np.ndarray:
    delta = seed_delta.copy()
    update_mask = mask & (~seed_mask)
    if not update_mask.any():
        return delta

    for _ in range(max(1, int(iterations))):
        numer = np.zeros_like(delta)
        denom = np.zeros(mask.shape, dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            src_y, dst_y = _shift_slice(mask.shape[0], dy)
            src_x, dst_x = _shift_slice(mask.shape[1], dx)
            valid = mask[dst_y, dst_x] & mask[src_y, src_x]
            if not valid.any():
                continue
            diff = luma[dst_y, dst_x] - luma[src_y, src_x]
            weight = np.exp(-edge_beta * diff * diff).astype(np.float32) * valid.astype(np.float32)
            numer[dst_y, dst_x] += weight[:, :, None] * delta[src_y, src_x]
            denom[dst_y, dst_x] += weight
        if anchor_strength > 0.0:
            denom[update_mask] += float(anchor_strength)
        updated = delta.copy()
        safe = update_mask & (denom > 1e-6)
        updated[safe] = numer[safe] / denom[safe, None]
        delta = updated
        delta[seed_mask] = seed_delta[seed_mask]
        delta[~mask] = 0.0
    return delta


def _apply_delta_to_base(
    *,
    base_rgb: np.ndarray,
    delta_ab: np.ndarray,
    feather: np.ndarray,
    strength: float,
    min_chroma: float,
) -> np.ndarray:
    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_ab = base_lab[:, :, 1:3] - 128.0
    final_ab = base_ab + (delta_ab * feather[:, :, None] * float(np.clip(strength, 0.0, 1.0)))
    magnitude = np.linalg.norm(final_ab, axis=2)
    need_boost = feather > 0.0
    floor = float(max(0.0, min_chroma))
    if floor > 0.0:
        scale = np.ones_like(magnitude)
        scale[need_boost] = np.maximum(1.0, floor / np.clip(magnitude[need_boost], 1e-3, None))
        final_ab = final_ab * scale[:, :, None]
    base_lab[:, :, 1:3] = np.clip(final_ab + 128.0, 0.0, 255.0)
    return cv2.cvtColor(base_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _select_palette_color_for_component(
    *,
    runtime: CostumePaletteRuntime,
    component_mask: np.ndarray,
    small_lab: np.ndarray,
    used_colors: list[object] | None = None,
) -> object:
    component_lab = small_lab[component_mask]
    mean_l = float(component_lab[:, 0].mean())
    mean_ab = component_lab[:, 1:3].mean(axis=0) - 128.0
    mean_chroma = float(np.linalg.norm(mean_ab))
    reliability = float(np.clip(mean_chroma / runtime.selection_chroma_reliability, 0.0, 1.0))
    available_palette = runtime.palette
    if used_colors and len(used_colors) < len(runtime.palette):
        used_ids = {id(color) for color in used_colors}
        remaining = [color for color in runtime.palette if id(color) not in used_ids]
        if remaining:
            available_palette = remaining
    best = available_palette[0]
    best_score = float("inf")
    for palette_color in available_palette:
        luminance_score = runtime.selection_luminance_weight * abs(mean_l - palette_color.lightness)
        chroma_score = runtime.selection_chroma_weight * reliability * float(np.linalg.norm(mean_ab - palette_color.ab))
        vividness_bonus = runtime.vividness_bias * palette_color.chroma
        warm_score = max(float(palette_color.ab[0]), 0.0) + (0.35 * max(float(palette_color.ab[1]), 0.0))
        distinct_penalty = 0.0
        for used_color in used_colors or []:
            distance = float(np.linalg.norm(palette_color.ab - used_color.ab))
            distinct_penalty += max(0.0, 24.0 - distance) * 0.35
        score = luminance_score + chroma_score + distinct_penalty - vividness_bonus - (runtime.warm_bias * warm_score)
        if score < best_score:
            best = palette_color
            best_score = score
    return best


def _pick_seed_points(component_mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    if not component_mask.any():
        return []
    work = component_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    interior = cv2.erode(work, kernel, iterations=1)
    if not interior.any():
        interior = work
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    seeds: list[tuple[int, int]] = []
    suppression = max(4, int(round(np.sqrt(float(component_mask.sum())) * 0.08)))
    for _ in range(max(1, count)):
        index = np.argmax(dist)
        seed_y, seed_x = np.unravel_index(index, dist.shape)
        if dist[seed_y, seed_x] <= 0.0:
            break
        seeds.append((int(seed_y), int(seed_x)))
        cv2.circle(dist, (int(seed_x), int(seed_y)), suppression, 0.0, thickness=-1)
    if not seeds:
        coords = np.column_stack(np.where(component_mask > 0))
        center = coords[len(coords) // 2]
        seeds.append((int(center[0]), int(center[1])))
    return seeds


def _select_components(
    mask: np.ndarray,
    *,
    min_component_fraction: float,
    max_components: int,
) -> tuple[np.ndarray, list[int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_pixels = max(1, int(round(mask.shape[0] * mask.shape[1] * float(min_component_fraction))))
    ranked = [
        (label, int(stats[label, cv2.CC_STAT_AREA]))
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return labels, [label for label, _ in ranked[:max_components]]


def _suppress_head_region(costume_mask: np.ndarray, person_mask: np.ndarray, skin_mask: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0 or not person_mask.any():
        return costume_mask.astype(np.uint8)
    suppressed = costume_mask.copy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_mask.astype(np.uint8), connectivity=8)
    yy = np.arange(person_mask.shape[0])[:, None]
    for label in range(1, num_labels):
        top = int(stats[label, cv2.CC_STAT_TOP])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        strong_fraction = max(0.28, fraction * 1.75)
        cutoff = min(person_mask.shape[0], top + max(1, int(round(height * strong_fraction))))
        suppressed[(labels == label) & (yy < cutoff)] = 0
        person_component = labels == label
        upper_component = person_component & (yy < min(person_mask.shape[0], top + max(1, int(round(height * 0.45)))))
        skin_component = (skin_mask > 0) & upper_component
        if skin_component.any():
            ys, xs = np.where(skin_component)
            center_x = float(xs.mean())
            center_y = float(ys.mean())
            radius_x = max(8.0, float(stats[label, cv2.CC_STAT_WIDTH]) * 0.30)
            radius_y = max(8.0, float(height) * 0.28)
            ellipse = (((np.arange(person_mask.shape[1])[None, :] - center_x) / radius_x) ** 2) + (
                ((np.arange(person_mask.shape[0])[:, None] - center_y) / radius_y) ** 2
            )
            suppressed[(ellipse <= 1.0) & person_component] = 0
    return suppressed


def _suppress_hair_like_regions(
    image_rgb: np.ndarray,
    costume_mask: np.ndarray,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    if not costume_mask.any():
        return costume_mask.astype(np.uint8)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]
    suppressed = costume_mask.copy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_mask.astype(np.uint8), connectivity=8)
    yy = np.arange(person_mask.shape[0])[:, None]
    xx = np.arange(person_mask.shape[1])[None, :]
    for label in range(1, num_labels):
        person_component = labels == label
        top = int(stats[label, cv2.CC_STAT_TOP])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        upper = person_component & (yy < min(person_mask.shape[0], top + max(1, int(round(height * 0.48)))))
        skin_upper = (skin_mask > 0) & upper
        if skin_upper.any():
            ys, xs = np.where(skin_upper)
            center_x = float(xs.mean())
            center_y = float(ys.mean())
        else:
            center_x = float(stats[label, cv2.CC_STAT_LEFT] + (width * 0.5))
            center_y = float(top + (height * 0.18))
        radius_x = max(10.0, width * 0.36)
        radius_y = max(10.0, height * 0.34)
        head_zone = ((((xx - center_x) / radius_x) ** 2) + (((yy - center_y) / radius_y) ** 2) <= 1.0) & person_component
        hair_like = (
            (suppressed > 0)
            & head_zone
            & (upper)
            & (value < 120)
            & (saturation < 90)
            & (lightness < 120)
        )
        suppressed[hair_like] = 0
    return suppressed


def _refine_component_masks(
    *,
    image_rgb: np.ndarray,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
    labels: np.ndarray,
    component_ids: list[int],
) -> list[np.ndarray]:
    refined: list[np.ndarray] = []
    all_costume = np.isin(labels, component_ids).astype(np.uint8)
    for component_id in component_ids:
        component_mask = (labels == component_id).astype(np.uint8)
        refined_mask = _refine_component_with_grabcut(
            image_rgb=image_rgb,
            component_mask=component_mask,
            all_costume_mask=all_costume,
            person_mask=person_mask,
            skin_mask=skin_mask,
        )
        if refined_mask.any():
            refined.append(refined_mask.astype(bool))
    return refined


def _refine_component_with_grabcut(
    *,
    image_rgb: np.ndarray,
    component_mask: np.ndarray,
    all_costume_mask: np.ndarray,
    person_mask: np.ndarray,
    skin_mask: np.ndarray,
) -> np.ndarray:
    if not component_mask.any():
        return component_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(component_mask, kernel, iterations=2)
    allowed_region = ((dilated > 0) & (person_mask > 0) & (skin_mask == 0)).astype(np.uint8)
    ys, xs = np.where(allowed_region > 0)
    if ys.size == 0 or xs.size == 0:
        return component_mask
    top = max(0, int(ys.min()) - 6)
    bottom = min(image_rgb.shape[0], int(ys.max()) + 7)
    left = max(0, int(xs.min()) - 6)
    right = min(image_rgb.shape[1], int(xs.max()) + 7)
    roi = image_rgb[top:bottom, left:right]
    roi_component = component_mask[top:bottom, left:right]
    roi_allowed = allowed_region[top:bottom, left:right]
    roi_skin = skin_mask[top:bottom, left:right]
    roi_other_costume = ((all_costume_mask[top:bottom, left:right] > 0) & (roi_component == 0)).astype(np.uint8)

    gc_mask = np.full(roi_component.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[roi_allowed == 0] = cv2.GC_BGD
    gc_mask[roi_skin > 0] = cv2.GC_BGD
    gc_mask[roi_component > 0] = cv2.GC_PR_FGD
    gc_mask[roi_other_costume > 0] = cv2.GC_PR_BGD
    upper_band = np.arange(roi_component.shape[0])[:, None] < max(1, int(round(roi_component.shape[0] * 0.22)))
    gc_mask[upper_band & (roi_component > 0)] = cv2.GC_BGD
    interior = cv2.erode(roi_component, kernel, iterations=1)
    if interior.any():
        gc_mask[interior > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
        refined_roi = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        refined_roi = roi_component.copy()

    refined_roi = refined_roi * roi_allowed
    refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_OPEN, kernel)
    refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_CLOSE, kernel)
    refined_roi = np.minimum(refined_roi, cv2.dilate(roi_component, kernel, iterations=2))
    refined = np.zeros_like(component_mask)
    refined[top:bottom, left:right] = refined_roi
    return refined.astype(np.uint8)


def _assign_palette_colors(
    *,
    runtime: CostumePaletteRuntime,
    component_masks: list[np.ndarray],
    small_lab: np.ndarray,
) -> list[object]:
    order = sorted(range(len(component_masks)), key=lambda idx: int(component_masks[idx].sum()), reverse=True)
    assigned: list[object | None] = [None] * len(component_masks)
    used_colors: list[object] = []
    for idx in order:
        color = _select_palette_color_for_component(
            runtime=runtime,
            component_mask=component_masks[idx],
            small_lab=small_lab,
            used_colors=used_colors,
        )
        assigned[idx] = color
        used_colors.append(color)
    return [color for color in assigned if color is not None]


def _build_anchor_mask(component_mask: np.ndarray) -> np.ndarray:
    if not component_mask.any():
        return component_mask
    area = float(component_mask.sum())
    radius = max(1, int(round(np.sqrt(area) * 0.03)))
    kernel_size = max(3, (radius * 2) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    anchor = cv2.erode(component_mask.astype(np.uint8), kernel, iterations=1)
    if anchor.any():
        return anchor
    return component_mask.astype(np.uint8)


def _build_target_field(
    *,
    component_mask: np.ndarray,
    anchor_mask: np.ndarray,
    base_ab: np.ndarray,
    target_ab: np.ndarray,
    preserve_local_variation: float,
    min_chroma: float,
) -> np.ndarray:
    target_field = base_ab.copy()
    component_values = base_ab[component_mask]
    component_chroma = np.linalg.norm(component_values, axis=1)
    reference_chroma = float(np.median(component_chroma)) if component_chroma.size else 1.0
    anchor_values = base_ab[anchor_mask]
    if anchor_values.size == 0:
        return target_field
    anchor_chroma = np.linalg.norm(anchor_values, axis=1)
    local_ratio = np.clip(anchor_chroma / max(reference_chroma, 1e-3), 0.70, 1.45)
    local_scale = (1.0 - preserve_local_variation) + (preserve_local_variation * local_ratio)
    scaled_target = target_ab[None, :] * local_scale[:, None]
    magnitude = np.linalg.norm(scaled_target, axis=1)
    floor = float(max(0.0, min_chroma))
    if floor > 0.0:
        scale = np.maximum(1.0, floor / np.clip(magnitude, 1e-3, None))
        scaled_target = scaled_target * scale[:, None]
    target_field[anchor_mask] = scaled_target
    return target_field


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


def _draw_seed_overlay(image_rgb: np.ndarray, seeds: list[CostumeSeed]) -> np.ndarray:
    overlay = image_rgb.copy()
    for seed in seeds:
        cv2.circle(overlay, (seed.x, seed.y), 8, tuple(int(channel) for channel in seed.rgb), thickness=-1)
        cv2.circle(overlay, (seed.x, seed.y), 10, (255, 255, 255), thickness=2)
    return overlay


def _build_mask_overlay(image_rgb: np.ndarray, person_mask: np.ndarray, skin_mask: np.ndarray, costume_mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    overlay[person_mask > 0] = overlay[person_mask > 0] * 0.70 + np.array([40, 120, 255], dtype=np.float32) * 0.30
    overlay[skin_mask > 0] = overlay[skin_mask > 0] * 0.55 + np.array([255, 140, 80], dtype=np.float32) * 0.45
    overlay[costume_mask > 0] = overlay[costume_mask > 0] * 0.45 + np.array([220, 60, 160], dtype=np.float32) * 0.55
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _resize_rgb(image_rgb: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return image_rgb.copy()
    height, width = image_rgb.shape[:2]
    return cv2.resize(
        image_rgb,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) <= 1e-3:
        return mask.copy()
    height, width = mask.shape[:2]
    resized = cv2.resize(
        mask.astype(np.uint8),
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    return (resized > 0).astype(np.uint8)


def _shift_slice(size: int, shift: int) -> tuple[slice, slice]:
    if shift > 0:
        return slice(0, size - shift), slice(shift, size)
    if shift < 0:
        return slice(-shift, size), slice(0, size + shift)
    return slice(0, size), slice(0, size)


def _rgb_to_pil(image_rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(image_rgb, mode="RGB")
