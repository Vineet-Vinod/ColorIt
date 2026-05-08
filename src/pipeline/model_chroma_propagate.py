from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer


def run_model_chroma_propagate(
    *,
    source_path: Path,
    model_color_path: Path,
    output_path: Path,
    keyframe_stride: int,
    chroma_blend: float,
    fallback_color_hex: str | None,
    fallback_strength: float,
    fallback_uncertainty: str,
    disagreement_start: float,
    disagreement_end: float,
    scene_cut_threshold: float,
    scene_keyframe_window: int,
    chroma_smooth_diameter: int,
    chroma_smooth_sigma_color: float,
    chroma_smooth_sigma_space: float,
    dark_fill_strength: float,
    dark_fill_luma_end: float,
    dark_fill_chroma_end: float,
    dark_fill_sigma: float,
    model_fill_strength: float,
    model_fill_chroma_end: float,
    model_fill_disagreement_start: float,
    model_fill_disagreement_end: float,
    model_fill_blur_sigma: float,
    model_fill_chroma_floor: float,
    component_fill_strength: float,
    component_fill_luma_end: float,
    component_fill_min_area: int,
    component_fill_model_chroma_min: float,
    blue_suppress_strength: float,
    blue_suppress_hue_start: float,
    blue_suppress_hue_end: float,
    semantic_consensus_manifest_path: Path | None,
    semantic_consensus_labels: list[str],
    semantic_protect_labels: list[str],
    semantic_protect_dilate: int,
    semantic_split_labels: list[str],
    semantic_consensus_strength: float,
    semantic_consensus_min_area: int,
    semantic_consensus_model_chroma_min: float,
    semantic_consensus_feather_sigma: float,
    semantic_consensus_diversify_strength: float,
    semantic_consensus_diversify_threshold: float,
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
    fallback_ab = _hex_to_lab_ab(fallback_color_hex) if fallback_color_hex else None

    source_info = ffprobe_media(source_path)
    model_info = ffprobe_media(model_color_path)
    width = int(source_info["width"])
    height = int(source_info["height"])
    if width != int(model_info["width"]) or height != int(model_info["height"]):
        raise ValueError("Source and model color clips must have matching dimensions.")

    source_frames = _read_frames(source_path, width=width, height=height)
    model_frames = _read_frames(model_color_path, width=width, height=height)
    frame_count = min(len(source_frames), len(model_frames))
    source_frames = source_frames[:frame_count]
    model_frames = model_frames[:frame_count]
    if frame_count == 0:
        raise ValueError("No frames available for chroma propagation.")
    semantic_masks_by_frame = _load_semantic_consensus_masks(
        manifest_path=semantic_consensus_manifest_path,
        labels=semantic_consensus_labels,
        protect_labels=semantic_protect_labels,
        protect_dilate=semantic_protect_dilate,
        split_labels=semantic_split_labels,
        frame_count=frame_count,
        width=width,
        height=height,
    )

    keyframe_indices = list(range(0, frame_count, max(1, keyframe_stride)))
    cut_indices = _detect_scene_cuts(
        source_frames=source_frames,
        threshold=scene_cut_threshold,
        keyframe_window=scene_keyframe_window,
    )
    keyframe_indices.extend(cut_indices)
    keyframe_indices = sorted(set(index for index in keyframe_indices if 0 <= index < frame_count))
    if keyframe_indices[-1] != frame_count - 1:
        keyframe_indices.append(frame_count - 1)
    output_frames = _propagate_chroma(
        source_frames=source_frames,
        model_frames=model_frames,
        keyframe_indices=keyframe_indices,
        chroma_blend=chroma_blend,
        fallback_ab=fallback_ab,
        fallback_strength=fallback_strength,
        fallback_uncertainty=fallback_uncertainty,
        disagreement_start=disagreement_start,
        disagreement_end=disagreement_end,
        chroma_smooth_diameter=chroma_smooth_diameter,
        chroma_smooth_sigma_color=chroma_smooth_sigma_color,
        chroma_smooth_sigma_space=chroma_smooth_sigma_space,
        dark_fill_strength=dark_fill_strength,
        dark_fill_luma_end=dark_fill_luma_end,
        dark_fill_chroma_end=dark_fill_chroma_end,
        dark_fill_sigma=dark_fill_sigma,
        model_fill_strength=model_fill_strength,
        model_fill_chroma_end=model_fill_chroma_end,
        model_fill_disagreement_start=model_fill_disagreement_start,
        model_fill_disagreement_end=model_fill_disagreement_end,
        model_fill_blur_sigma=model_fill_blur_sigma,
        model_fill_chroma_floor=model_fill_chroma_floor,
        component_fill_strength=component_fill_strength,
        component_fill_luma_end=component_fill_luma_end,
        component_fill_min_area=component_fill_min_area,
        component_fill_model_chroma_min=component_fill_model_chroma_min,
        blue_suppress_strength=blue_suppress_strength,
        blue_suppress_hue_start=blue_suppress_hue_start,
        blue_suppress_hue_end=blue_suppress_hue_end,
        semantic_masks_by_frame=semantic_masks_by_frame,
        semantic_consensus_strength=semantic_consensus_strength,
        semantic_consensus_min_area=semantic_consensus_min_area,
        semantic_consensus_model_chroma_min=semantic_consensus_model_chroma_min,
        semantic_consensus_feather_sigma=semantic_consensus_feather_sigma,
        semantic_consensus_diversify_strength=semantic_consensus_diversify_strength,
        semantic_consensus_diversify_threshold=semantic_consensus_diversify_threshold,
    )
    _write_frames(
        output_path=output_path,
        frames=output_frames,
        width=width,
        height=height,
        fps=str(source_info["fps"]),
        audio_input_path=source_path,
    )
    print(f"Model chroma propagation written: {output_path}")
    print(f"Frames: {frame_count}")
    print(f"Keyframes: {len(keyframe_indices)}")
    print(f"Scene-cut keyframes: {len(cut_indices)}")
    return 0


def _propagate_chroma(
    *,
    source_frames: list[np.ndarray],
    model_frames: list[np.ndarray],
    keyframe_indices: list[int],
    chroma_blend: float,
    fallback_ab: np.ndarray | None,
    fallback_strength: float,
    fallback_uncertainty: str,
    disagreement_start: float,
    disagreement_end: float,
    chroma_smooth_diameter: int,
    chroma_smooth_sigma_color: float,
    chroma_smooth_sigma_space: float,
    dark_fill_strength: float,
    dark_fill_luma_end: float,
    dark_fill_chroma_end: float,
    dark_fill_sigma: float,
    model_fill_strength: float,
    model_fill_chroma_end: float,
    model_fill_disagreement_start: float,
    model_fill_disagreement_end: float,
    model_fill_blur_sigma: float,
    model_fill_chroma_floor: float,
    component_fill_strength: float,
    component_fill_luma_end: float,
    component_fill_min_area: int,
    component_fill_model_chroma_min: float,
    blue_suppress_strength: float,
    blue_suppress_hue_start: float,
    blue_suppress_hue_end: float,
    semantic_masks_by_frame: list[list[np.ndarray]],
    semantic_consensus_strength: float,
    semantic_consensus_min_area: int,
    semantic_consensus_model_chroma_min: float,
    semantic_consensus_feather_sigma: float,
    semantic_consensus_diversify_strength: float,
    semantic_consensus_diversify_threshold: float,
) -> list[np.ndarray]:
    if fallback_uncertainty not in {"ab-delta", "hue"}:
        raise ValueError(f"Unsupported fallback uncertainty mode: {fallback_uncertainty}")
    source_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in source_frames]
    source_l = [cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, :1].astype(np.float32) for frame in source_frames]
    model_ab_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 1:3].astype(np.float32) for frame in model_frames]
    model_ab_by_key = {index: model_ab_frames[index] for index in keyframe_indices}
    semantic_consensus_by_frame = _build_temporal_semantic_consensus(
        masks_by_frame=semantic_masks_by_frame,
        model_ab_frames=model_ab_frames,
        strength=semantic_consensus_strength,
        min_area=semantic_consensus_min_area,
        model_chroma_min=semantic_consensus_model_chroma_min,
        diversify_strength=semantic_consensus_diversify_strength,
        diversify_threshold=semantic_consensus_diversify_threshold,
    )

    forward_ab: dict[int, np.ndarray] = {}
    for start, end in zip(keyframe_indices, keyframe_indices[1:]):
        ab = model_ab_by_key[start]
        forward_ab[start] = ab
        for frame_index in range(start + 1, end + 1):
            ab = _warp_ab(previous_ab=ab, previous_gray=source_gray[frame_index - 1], current_gray=source_gray[frame_index])
            forward_ab[frame_index] = ab

    backward_ab: dict[int, np.ndarray] = {}
    for start, end in zip(reversed(keyframe_indices[:-1]), reversed(keyframe_indices[1:])):
        ab = model_ab_by_key[end]
        backward_ab[end] = ab
        for frame_index in range(end - 1, start - 1, -1):
            ab = _warp_ab(previous_ab=ab, previous_gray=source_gray[frame_index + 1], current_gray=source_gray[frame_index])
            backward_ab[frame_index] = ab

    output_frames: list[np.ndarray] = []
    blend = float(np.clip(chroma_blend, 0.0, 1.0))
    for frame_index in range(len(source_frames)):
        left_key = max(index for index in keyframe_indices if index <= frame_index)
        right_key = min(index for index in keyframe_indices if index >= frame_index)
        if left_key == right_key:
            propagated_ab = model_ab_by_key[left_key]
            disagreement = None
        else:
            t = (frame_index - left_key) / max(right_key - left_key, 1)
            forward = forward_ab[frame_index]
            backward = backward_ab[frame_index]
            propagated_ab = (1.0 - t) * forward + t * backward
            if fallback_ab is not None and fallback_strength > 0.0:
                disagreement = _chroma_disagreement(forward=forward, backward=backward, mode=fallback_uncertainty)
                uncertainty = _smoothstep(disagreement_start, disagreement_end, disagreement)
                uncertainty = cv2.GaussianBlur(uncertainty.astype(np.float32), (0, 0), 2.0)
                fallback_mix = np.clip(uncertainty * fallback_strength, 0.0, 1.0)[:, :, None]
                propagated_ab = (1.0 - fallback_mix) * propagated_ab + fallback_mix * fallback_ab
            else:
                disagreement = _chroma_disagreement(forward=forward, backward=backward, mode=fallback_uncertainty)
        model_ab = model_ab_frames[frame_index]
        propagated_ab = _fill_from_model_chroma(
            propagated_ab,
            model_ab=model_ab,
            disagreement=disagreement,
            strength=model_fill_strength,
            chroma_end=model_fill_chroma_end,
            disagreement_start=model_fill_disagreement_start,
            disagreement_end=model_fill_disagreement_end,
            blur_sigma=model_fill_blur_sigma,
            chroma_floor=model_fill_chroma_floor,
        )
        propagated_ab = _fill_dark_components_from_model(
            propagated_ab,
            model_ab=model_ab,
            source_l=source_l[frame_index],
            strength=component_fill_strength,
            luma_end=component_fill_luma_end,
            min_area=component_fill_min_area,
            model_chroma_min=component_fill_model_chroma_min,
        )
        propagated_ab = _apply_semantic_chroma_consensus(
            propagated_ab,
            targets=semantic_consensus_by_frame[frame_index],
            strength=semantic_consensus_strength,
            feather_sigma=semantic_consensus_feather_sigma,
        )
        propagated_ab = _smooth_ab(
            propagated_ab,
            diameter=chroma_smooth_diameter,
            sigma_color=chroma_smooth_sigma_color,
            sigma_space=chroma_smooth_sigma_space,
        )
        propagated_ab = _fill_dark_low_chroma(
            propagated_ab,
            source_l=source_l[frame_index],
            strength=dark_fill_strength,
            luma_end=dark_fill_luma_end,
            chroma_end=dark_fill_chroma_end,
            sigma=dark_fill_sigma,
        )
        if fallback_ab is not None and blue_suppress_strength > 0.0:
            propagated_ab = _suppress_blue_ab(
                propagated_ab,
                replacement_ab=fallback_ab,
                strength=blue_suppress_strength,
                hue_start=blue_suppress_hue_start,
                hue_end=blue_suppress_hue_end,
            )
        deoldify_ab = cv2.cvtColor(source_frames[frame_index], cv2.COLOR_RGB2LAB)[:, :, 1:3].astype(np.float32)
        output_ab = (1.0 - blend) * deoldify_ab + blend * propagated_ab
        output_lab = np.concatenate([source_l[frame_index], output_ab], axis=2)
        output_rgb = cv2.cvtColor(np.clip(output_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        output_frames.append(output_rgb)
    return output_frames


def _load_semantic_consensus_masks(
    *,
    manifest_path: Path | None,
    labels: list[str],
    protect_labels: list[str],
    protect_dilate: int,
    split_labels: list[str],
    frame_count: int,
    width: int,
    height: int,
) -> list[list[np.ndarray]]:
    masks_by_frame: list[list[np.ndarray]] = [[] for _ in range(frame_count)]
    if manifest_path is None:
        return masks_by_frame
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent
    label_set = {label.strip() for label in labels if label.strip()}
    protect_label_set = {label.strip() for label in protect_labels if label.strip()}
    split_label_set = {label.strip() for label in split_labels if label.strip()}
    protect_masks_by_frame: list[np.ndarray] = [np.zeros((height, width), dtype=bool) for _ in range(frame_count)]
    split_centers_by_frame: list[list[tuple[float, float]]] = [[] for _ in range(frame_count)]
    for frame in manifest.get("frames", [])[:frame_count]:
        frame_index = int(frame["frame_index"])
        if frame_index >= frame_count:
            continue
        for instance in frame.get("instances", []):
            label = instance.get("label")
            if label_set and label not in label_set and label not in protect_label_set and label not in split_label_set:
                continue
            mask_path = manifest_dir / str(instance["mask_path"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Semantic mask not found: {mask_path}")
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            bool_mask = mask >= 128
            if label in protect_label_set:
                protect_masks_by_frame[frame_index] |= bool_mask
            elif label in split_label_set:
                split_centers_by_frame[frame_index].extend(_mask_centers(bool_mask, min_area=120))
            elif not label_set or label in label_set:
                masks_by_frame[frame_index].append(bool_mask)
    kernel = None
    if protect_label_set and protect_dilate > 0:
        size = protect_dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    for frame_index, masks in enumerate(masks_by_frame):
        protect_mask = protect_masks_by_frame[frame_index].astype(np.uint8)
        if kernel is not None:
            protect_mask = cv2.dilate(protect_mask, kernel, iterations=1)
        protect_bool = protect_mask > 0
        cleaned_masks = [mask & ~protect_bool for mask in masks]
        masks_by_frame[frame_index] = _split_masks_by_centers(cleaned_masks, split_centers_by_frame[frame_index])
    return masks_by_frame


def _mask_centers(mask: np.ndarray, *, min_area: int) -> list[tuple[float, float]]:
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    centers: list[tuple[float, float]] = []
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) >= min_area:
            centers.append((float(centroids[component_index][0]), float(centroids[component_index][1])))
    return centers


def _split_masks_by_centers(masks: list[np.ndarray], centers: list[tuple[float, float]]) -> list[np.ndarray]:
    if len(centers) < 2:
        return masks
    center_array = np.array(centers, dtype=np.float32)
    split_masks: list[np.ndarray] = []
    for mask in masks:
        y_coords, x_coords = np.nonzero(mask)
        if len(x_coords) == 0:
            continue
        points = np.stack([x_coords, y_coords], axis=1).astype(np.float32)
        distances = np.linalg.norm(points[:, None, :] - center_array[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        for center_index in range(len(centers)):
            selected = assignments == center_index
            if int(np.count_nonzero(selected)) < 200:
                continue
            split_mask = np.zeros_like(mask, dtype=bool)
            split_mask[y_coords[selected], x_coords[selected]] = True
            split_masks.append(split_mask)
    return split_masks


def _build_temporal_semantic_consensus(
    *,
    masks_by_frame: list[list[np.ndarray]],
    model_ab_frames: list[np.ndarray],
    strength: float,
    min_area: int,
    model_chroma_min: float,
    diversify_strength: float,
    diversify_threshold: float,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    consensus_by_frame: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in masks_by_frame]
    if strength <= 0.0 or not any(masks_by_frame):
        return consensus_by_frame

    tracks: dict[int, dict[str, object]] = {}
    next_track_id = 1
    component_refs: list[tuple[int, int, np.ndarray]] = []
    cooccurring_track_ids_by_frame: list[list[int]] = []

    for frame_index, masks in enumerate(masks_by_frame):
        model_ab = model_ab_frames[frame_index]
        model_chroma = np.linalg.norm(model_ab - 128.0, axis=2)
        components = _extract_semantic_components(
            masks=masks,
            model_ab=model_ab,
            model_chroma=model_chroma,
            min_area=min_area,
            model_chroma_min=model_chroma_min,
        )
        assigned_tracks: set[int] = set()
        new_active_track_ids: list[int] = []
        active_track_ids = [
            track_id
            for track_id, track in tracks.items()
            if frame_index - int(track.get("last_frame_index", -9999)) <= 8
        ]
        for component in components:
            track_id = _match_semantic_track(
                component=component,
                tracks=tracks,
                active_track_ids=[track_id for track_id in active_track_ids if track_id not in assigned_tracks],
            )
            if track_id is None:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = {"samples": []}
            assigned_tracks.add(track_id)
            new_active_track_ids.append(track_id)
            tracks[track_id]["bbox"] = component["bbox"]
            tracks[track_id]["centroid"] = component["centroid"]
            tracks[track_id]["last_frame_index"] = frame_index
            tracks[track_id].setdefault("centroids", []).append(component["centroid"])
            tracks[track_id]["samples"].append(component["samples"])
            component_refs.append((frame_index, track_id, component["mask"]))
        cooccurring_track_ids_by_frame.append(new_active_track_ids)

    track_targets: dict[int, np.ndarray] = {}
    for track_id, track in tracks.items():
        samples = track["samples"]
        if not samples:
            continue
        track_targets[track_id] = np.median(np.concatenate(samples, axis=0), axis=0).astype(np.float32)
    if diversify_strength > 0.0:
        _diversify_cooccurring_track_targets(
            track_targets=track_targets,
            tracks=tracks,
            cooccurring_track_ids_by_frame=cooccurring_track_ids_by_frame,
            strength=diversify_strength,
            threshold=diversify_threshold,
        )

    for frame_index, track_id, mask in component_refs:
        target_ab = track_targets.get(track_id)
        if target_ab is not None:
            consensus_by_frame[frame_index].append((mask, target_ab))
    return consensus_by_frame


def _extract_semantic_components(
    *,
    masks: list[np.ndarray],
    model_ab: np.ndarray,
    model_chroma: np.ndarray,
    min_area: int,
    model_chroma_min: float,
) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for mask in masks:
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component_mask = labels == component_index
            confident_mask = component_mask & (model_chroma >= model_chroma_min)
            confident_count = int(np.count_nonzero(confident_mask))
            if confident_count < max(20, min_area // 20):
                continue
            samples = model_ab[confident_mask]
            sample_step = max(1, len(samples) // 500)
            components.append(
                {
                    "mask": component_mask,
                    "bbox": [
                        int(stats[component_index, cv2.CC_STAT_LEFT]),
                        int(stats[component_index, cv2.CC_STAT_TOP]),
                        int(stats[component_index, cv2.CC_STAT_WIDTH]),
                        int(stats[component_index, cv2.CC_STAT_HEIGHT]),
                    ],
                    "centroid": (float(centroids[component_index][0]), float(centroids[component_index][1])),
                    "samples": samples[::sample_step].astype(np.float32),
                }
            )
    return components


def _match_semantic_track(
    *,
    component: dict[str, object],
    tracks: dict[int, dict[str, object]],
    active_track_ids: list[int],
) -> int | None:
    best_track_id: int | None = None
    best_score = 0.0
    for track_id in active_track_ids:
        track = tracks[track_id]
        iou = _bbox_iou(component["bbox"], track.get("bbox"))
        distance = _centroid_distance(component["centroid"], track.get("centroid"))
        score = iou + max(0.0, 1.0 - distance / 140.0)
        if score > best_score:
            best_score = score
            best_track_id = track_id
    if best_score < 0.25:
        return None
    return best_track_id


def _diversify_cooccurring_track_targets(
    *,
    track_targets: dict[int, np.ndarray],
    tracks: dict[int, dict[str, object]],
    cooccurring_track_ids_by_frame: list[list[int]],
    strength: float,
    threshold: float,
) -> None:
    close_edges: set[tuple[int, int]] = set()
    for track_ids in cooccurring_track_ids_by_frame:
        visible = [track_id for track_id in track_ids if track_id in track_targets]
        for first_index, first_id in enumerate(visible):
            for second_id in visible[first_index + 1 :]:
                distance = float(np.linalg.norm(track_targets[first_id] - track_targets[second_id]))
                if distance < threshold:
                    close_edges.add(tuple(sorted((first_id, second_id))))
    if not close_edges:
        return

    palette = np.array(
        [
            _hex_to_lab_ab("#556f9f"),
            _hex_to_lab_ab("#815a82"),
            _hex_to_lab_ab("#777446"),
            _hex_to_lab_ab("#4f7d78"),
            _hex_to_lab_ab("#89525a"),
        ],
        dtype=np.float32,
    )
    involved_track_ids = sorted({track_id for edge in close_edges for track_id in edge}, key=lambda track_id: _track_x(tracks, track_id))
    used_palette_indices: set[int] = set()
    for order, track_id in enumerate(involved_track_ids):
        original = track_targets[track_id]
        palette_index = order % len(palette)
        used_palette_indices.add(palette_index)
        track_targets[track_id] = (1.0 - strength) * original + strength * palette[palette_index]


def _track_x(tracks: dict[int, dict[str, object]], track_id: int) -> float:
    centroids = tracks[track_id].get("centroids", [])
    if not centroids:
        return 0.0
    return float(np.median([centroid[0] for centroid in centroids]))


def _bbox_iou(first: object, second: object) -> float:
    if second is None:
        return 0.0
    x1, y1, w1, h1 = [float(value) for value in first]
    x2, y2, w2, h2 = [float(value) for value in second]
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / max(union, 1e-6)


def _centroid_distance(first: object, second: object) -> float:
    if second is None:
        return float("inf")
    x1, y1 = [float(value) for value in first]
    x2, y2 = [float(value) for value in second]
    return float(np.hypot(x1 - x2, y1 - y2))


def _apply_semantic_chroma_consensus(
    ab: np.ndarray,
    *,
    targets: list[tuple[np.ndarray, np.ndarray]],
    strength: float,
    feather_sigma: float,
) -> np.ndarray:
    if strength <= 0.0 or not targets:
        return ab
    output = ab.copy()
    for component_mask, target_ab in targets:
        mix_mask = component_mask.astype(np.float32)
        if feather_sigma > 0.0:
            mix_mask = cv2.GaussianBlur(mix_mask, (0, 0), feather_sigma)
        mix = np.clip(mix_mask * strength, 0.0, 1.0)[:, :, None]
        output = (1.0 - mix) * output + mix * target_ab
    return output


def _detect_scene_cuts(*, source_frames: list[np.ndarray], threshold: float, keyframe_window: int) -> list[int]:
    if threshold <= 0.0 or len(source_frames) < 2:
        return []
    previous_gray = cv2.cvtColor(source_frames[0], cv2.COLOR_RGB2GRAY)
    cut_indices: set[int] = set()
    for frame_index, frame in enumerate(source_frames[1:], start=1):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_delta = float(np.mean(cv2.absdiff(previous_gray, current_gray)))
        if mean_delta >= threshold:
            for offset in range(-keyframe_window, keyframe_window + 1):
                cut_indices.add(frame_index + offset)
        previous_gray = current_gray
    return sorted(cut_indices)


def _smooth_ab(ab: np.ndarray, *, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if diameter <= 0:
        return ab
    diameter = diameter if diameter % 2 == 1 else diameter + 1
    channels = [
        cv2.bilateralFilter(
            ab[:, :, channel].astype(np.float32),
            diameter,
            sigma_color,
            sigma_space,
        )
        for channel in range(2)
    ]
    return np.stack(channels, axis=2)


def _fill_from_model_chroma(
    ab: np.ndarray,
    *,
    model_ab: np.ndarray,
    disagreement: np.ndarray | None,
    strength: float,
    chroma_end: float,
    disagreement_start: float,
    disagreement_end: float,
    blur_sigma: float,
    chroma_floor: float,
) -> np.ndarray:
    if strength <= 0.0:
        return ab
    propagated_chroma = np.linalg.norm(ab - 128.0, axis=2)
    model_chroma = np.linalg.norm(model_ab - 128.0, axis=2)
    model_fill_ab = model_ab
    if chroma_floor > 0.0:
        centered = model_ab - 128.0
        chroma_safe = np.maximum(model_chroma, 1e-3)
        boosted_chroma = np.maximum(model_chroma, chroma_floor)
        model_fill_ab = 128.0 + centered * (boosted_chroma / chroma_safe)[:, :, None]
    weak_mask = 1.0 - _smoothstep(max(chroma_end - 16.0, 0.0), chroma_end, propagated_chroma)
    model_confident = _smoothstep(max(chroma_end - 18.0, 0.0), chroma_end + 18.0, model_chroma)
    if disagreement is None:
        uncertainty = weak_mask
    else:
        uncertainty = np.maximum(
            weak_mask,
            _smoothstep(disagreement_start, disagreement_end, disagreement),
        )
    mask = uncertainty * model_confident
    if blur_sigma > 0.0:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), blur_sigma)
    mix = np.clip(mask * strength, 0.0, 1.0)[:, :, None]
    return (1.0 - mix) * ab + mix * model_fill_ab


def _fill_dark_components_from_model(
    ab: np.ndarray,
    *,
    model_ab: np.ndarray,
    source_l: np.ndarray,
    strength: float,
    luma_end: float,
    min_area: int,
    model_chroma_min: float,
) -> np.ndarray:
    if strength <= 0.0:
        return ab
    dark_mask = (source_l[:, :, 0] <= luma_end).astype(np.uint8)
    if not np.any(dark_mask):
        return ab
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    output = ab.copy()
    model_centered = model_ab - 128.0
    model_chroma = np.linalg.norm(model_centered, axis=2)
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component_mask = labels == component_index
        confident_mask = component_mask & (model_chroma >= model_chroma_min)
        if int(np.count_nonzero(confident_mask)) < max(25, min_area // 30):
            continue
        target_ab = np.median(model_ab[confident_mask], axis=0).astype(np.float32)
        component_mix = np.zeros(labels.shape, dtype=np.float32)
        component_mix[component_mask] = strength
        component_mix = cv2.GaussianBlur(component_mix, (0, 0), 1.2)
        mix = np.clip(component_mix, 0.0, 1.0)[:, :, None]
        output = (1.0 - mix) * output + mix * target_ab
    return output


def _fill_dark_low_chroma(
    ab: np.ndarray,
    *,
    source_l: np.ndarray,
    strength: float,
    luma_end: float,
    chroma_end: float,
    sigma: float,
) -> np.ndarray:
    if strength <= 0.0 or sigma <= 0.0:
        return ab
    chroma = np.linalg.norm(ab - 128.0, axis=2)
    confident = _smoothstep(chroma_end, chroma_end + 28.0, chroma)
    blurred_weight = cv2.GaussianBlur(confident.astype(np.float32), (0, 0), sigma)
    centered = (ab - 128.0) * confident[:, :, None]
    borrowed = cv2.GaussianBlur(centered.astype(np.float32), (0, 0), sigma)
    borrowed = borrowed / np.maximum(blurred_weight[:, :, None], 1e-3) + 128.0
    borrowed = np.where(blurred_weight[:, :, None] > 0.05, borrowed, ab)
    dark_mask = 1.0 - _smoothstep(max(luma_end - 35.0, 0.0), luma_end, source_l[:, :, 0])
    low_chroma_mask = 1.0 - _smoothstep(max(chroma_end - 14.0, 0.0), chroma_end, chroma)
    mask = cv2.GaussianBlur((dark_mask * low_chroma_mask).astype(np.float32), (0, 0), 1.5)
    mix = np.clip(mask * strength, 0.0, 1.0)[:, :, None]
    return (1.0 - mix) * ab + mix * borrowed


def _suppress_blue_ab(
    ab: np.ndarray,
    *,
    replacement_ab: np.ndarray,
    strength: float,
    hue_start: float,
    hue_end: float,
) -> np.ndarray:
    lab = np.empty((ab.shape[0], ab.shape[1], 3), dtype=np.uint8)
    lab[:, :, 0] = 128
    lab[:, :, 1:3] = np.clip(ab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    if hue_start <= hue_end:
        hue_mask = (hue >= hue_start) & (hue <= hue_end)
    else:
        hue_mask = (hue >= hue_start) | (hue <= hue_end)
    chroma = np.linalg.norm(ab - 128.0, axis=2)
    hue_mask = hue_mask.astype(np.float32) * _smoothstep(35.0, 95.0, saturation)
    cool_lab_mask = _smoothstep(6.0, 28.0, 128.0 - ab[:, :, 1])
    cool_purple_mask = _smoothstep(8.0, 28.0, ab[:, :, 0] - 128.0)
    cool_purple_mask *= 1.0 - _smoothstep(12.0, 34.0, ab[:, :, 1] - 128.0)
    mask = np.maximum(np.maximum(hue_mask, cool_lab_mask), cool_purple_mask)
    mask *= _smoothstep(10.0, 35.0, chroma)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    mix = np.clip(mask * strength, 0.0, 1.0)[:, :, None]
    return (1.0 - mix) * ab + mix * replacement_ab


def _chroma_disagreement(*, forward: np.ndarray, backward: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ab-delta":
        return np.linalg.norm(forward - backward, axis=2)

    forward_centered = forward - 128.0
    backward_centered = backward - 128.0
    forward_norm = np.linalg.norm(forward_centered, axis=2)
    backward_norm = np.linalg.norm(backward_centered, axis=2)
    norm_product = np.maximum(forward_norm * backward_norm, 1e-6)
    cosine = np.sum(forward_centered * backward_centered, axis=2) / norm_product
    hue_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    low_chroma = 1.0 - _smoothstep(8.0, 24.0, np.minimum(forward_norm, backward_norm))
    return np.maximum(hue_angle, low_chroma * 45.0)


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    value = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _hex_to_lab_ab(value: str | None) -> np.ndarray:
    if value is None:
        raise ValueError("Expected a #RRGGBB fallback color.")
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB fallback color, got: {value}")
    rgb = np.array([[[int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)]]], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    return lab[1:3]


def _warp_ab(*, previous_ab: np.ndarray, previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    current_to_previous_flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        previous_gray,
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.2,
        0,
    )
    height, width = current_gray.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + current_to_previous_flow[:, :, 0]
    map_y = grid_y + current_to_previous_flow[:, :, 1]
    channels = [
        cv2.remap(
            previous_ab[:, :, channel],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for channel in range(2)
    ]
    return np.stack(channels, axis=2)


def _read_frames(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    frames: list[np.ndarray] = []
    try:
        while True:
            frame_data = reader.stdout.read(frame_bytes)
            if not frame_data:
                break
            if len(frame_data) != frame_bytes:
                raise RuntimeError(f"Unexpected rawvideo frame size: {len(frame_data)}")
            frames.append(np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3)).copy())
        returncode = reader.wait()
    finally:
        if reader.stdout is not None:
            reader.stdout.close()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo reader failed: {reader.stderr.read().decode().strip()}")
    if reader.stderr is not None:
        reader.stderr.close()
    return frames


def _write_frames(
    *,
    output_path: Path,
    frames: list[np.ndarray],
    width: int,
    height: int,
    fps: str,
    audio_input_path: Path,
) -> None:
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
    if writer.stdin is None or writer.stderr is None:
        raise RuntimeError("ffmpeg rawvideo writer failed to expose stdin/stderr pipes.")
    try:
        for frame in frames:
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
        writer.stdin.close()
        returncode = writer.wait()
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg rawvideo writer failed: {writer.stderr.read().decode().strip()}")
    writer.stderr.close()
