from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.ffmpeg_utils import ffprobe_media, open_rawvideo_reader, open_rawvideo_writer


SEMANTIC_PROTECT_DILATE = 3
SEMANTIC_MIN_AREA = 600
SEMANTIC_MODEL_CHROMA_MIN = 8.0
SEMANTIC_FEATHER_SIGMA = 1.2
SEMANTIC_DIVERSIFY_STRENGTH = 0.65
SEMANTIC_DIVERSIFY_THRESHOLD = 20.0


def run_model_chroma_propagate(
    *,
    source_path: Path,
    model_color_path: Path,
    output_path: Path,
    semantic_consensus_manifest_path: Path,
    semantic_consensus_labels: list[str],
    semantic_protect_labels: list[str],
    semantic_split_labels: list[str],
    semantic_consensus_strength: float,
    output_preset: str,
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

    source_info = ffprobe_media(source_path)
    model_info = ffprobe_media(model_color_path)
    width = int(source_info["width"])
    height = int(source_info["height"])
    if width != int(model_info["width"]) or height != int(model_info["height"]):
        raise ValueError("Source and model color clips must have matching dimensions.")

    source_frames = _read_frames(source_path, width=width, height=height)
    model_frames = _read_frames(model_color_path, width=width, height=height)
    frame_count = min(len(source_frames), len(model_frames))
    if frame_count == 0:
        raise ValueError("No frames available for chroma propagation.")

    source_frames = source_frames[:frame_count]
    model_frames = model_frames[:frame_count]
    masks_by_frame = _load_semantic_consensus_masks(
        manifest_path=semantic_consensus_manifest_path,
        labels=semantic_consensus_labels,
        protect_labels=semantic_protect_labels,
        split_labels=semantic_split_labels,
        frame_count=frame_count,
        width=width,
        height=height,
    )
    output_frames = _apply_model_chroma_consensus(
        source_frames=source_frames,
        model_frames=model_frames,
        masks_by_frame=masks_by_frame,
        semantic_consensus_strength=semantic_consensus_strength,
    )
    _write_frames(
        output_path=output_path,
        frames=output_frames,
        width=width,
        height=height,
        fps=str(source_info["fps"]),
        preset=output_preset,
        audio_input_path=source_path,
    )
    print(f"Model chroma consensus written: {output_path}")
    print(f"Frames: {frame_count}")
    return 0


def _apply_model_chroma_consensus(
    *,
    source_frames: list[np.ndarray],
    model_frames: list[np.ndarray],
    masks_by_frame: list[list[np.ndarray]],
    semantic_consensus_strength: float,
) -> list[np.ndarray]:
    source_l = [cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, :1].astype(np.float32) for frame in source_frames]
    model_ab_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 1:3].astype(np.float32) for frame in model_frames]
    consensus_by_frame = _build_temporal_semantic_consensus(
        masks_by_frame=masks_by_frame,
        model_ab_frames=model_ab_frames,
        strength=semantic_consensus_strength,
        min_area=SEMANTIC_MIN_AREA,
        model_chroma_min=SEMANTIC_MODEL_CHROMA_MIN,
        diversify_strength=SEMANTIC_DIVERSIFY_STRENGTH,
        diversify_threshold=SEMANTIC_DIVERSIFY_THRESHOLD,
    )

    output_frames: list[np.ndarray] = []
    mix_cache: dict[int, np.ndarray] = {}
    for frame_index, model_ab in enumerate(model_ab_frames):
        output_ab = _apply_semantic_chroma_consensus(
            model_ab,
            targets=consensus_by_frame[frame_index],
            strength=semantic_consensus_strength,
            feather_sigma=SEMANTIC_FEATHER_SIGMA,
            mix_cache=mix_cache,
        )
        output_lab = np.concatenate([source_l[frame_index], output_ab], axis=2)
        output_frames.append(cv2.cvtColor(np.clip(output_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    return output_frames


def _load_semantic_consensus_masks(
    *,
    manifest_path: Path,
    labels: list[str],
    protect_labels: list[str],
    split_labels: list[str],
    frame_count: int,
    width: int,
    height: int,
) -> list[list[np.ndarray]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent
    label_set = {label.strip() for label in labels if label.strip()}
    protect_label_set = {label.strip() for label in protect_labels if label.strip()}
    split_label_set = {label.strip() for label in split_labels if label.strip()}
    masks_by_frame: list[list[np.ndarray]] = [[] for _ in range(frame_count)]
    protect_masks_by_frame = [np.zeros((height, width), dtype=bool) for _ in range(frame_count)]
    split_centers_by_frame: list[list[tuple[float, float]]] = [[] for _ in range(frame_count)]
    mask_cache: dict[Path, np.ndarray] = {}

    for frame in manifest.get("frames", [])[:frame_count]:
        frame_index = int(frame["frame_index"])
        if frame_index >= frame_count:
            continue
        for instance in frame.get("instances", []):
            label = instance.get("label")
            if label_set and label not in label_set and label not in protect_label_set and label not in split_label_set:
                continue
            mask_path = manifest_dir / str(instance["mask_path"])
            bool_mask = _read_bool_mask(mask_path=mask_path, width=width, height=height, cache=mask_cache)
            if label in protect_label_set:
                protect_masks_by_frame[frame_index] |= bool_mask
            elif label in split_label_set:
                split_centers_by_frame[frame_index].extend(_mask_centers(bool_mask, min_area=120))
            elif not label_set or label in label_set:
                masks_by_frame[frame_index].append(bool_mask)

    kernel = None
    if protect_label_set and SEMANTIC_PROTECT_DILATE > 0:
        size = SEMANTIC_PROTECT_DILATE * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    for frame_index, masks in enumerate(masks_by_frame):
        protect_mask = protect_masks_by_frame[frame_index].astype(np.uint8)
        if kernel is not None:
            protect_mask = cv2.dilate(protect_mask, kernel, iterations=1)
        masks_by_frame[frame_index] = _split_masks_by_centers(
            [mask & ~(protect_mask > 0) for mask in masks],
            split_centers_by_frame[frame_index],
        )
    return masks_by_frame


def _read_bool_mask(*, mask_path: Path, width: int, height: int, cache: dict[Path, np.ndarray]) -> np.ndarray:
    mask = cache.get(mask_path)
    if mask is None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Semantic mask not found: {mask_path}")
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cache[mask_path] = mask >= 128
    return cache[mask_path]


def _mask_centers(mask: np.ndarray, *, min_area: int) -> list[tuple[float, float]]:
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [
        (float(centroids[index][0]), float(centroids[index][1]))
        for index in range(1, component_count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
    ]


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
        assignments = np.argmin(np.linalg.norm(points[:, None, :] - center_array[None, :, :], axis=2), axis=1)
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
    component_geometry_cache: dict[int, list[dict[str, object]]] = {}

    for frame_index, masks in enumerate(masks_by_frame):
        model_ab = model_ab_frames[frame_index]
        components = _extract_semantic_components(
            masks=masks,
            model_ab=model_ab,
            model_chroma=np.linalg.norm(model_ab - 128.0, axis=2),
            min_area=min_area,
            model_chroma_min=model_chroma_min,
            geometry_cache=component_geometry_cache,
        )
        assigned_tracks: set[int] = set()
        active_track_ids = [
            track_id
            for track_id, track in tracks.items()
            if frame_index - int(track.get("last_frame_index", -9999)) <= 8
        ]
        visible_track_ids: list[int] = []
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
            visible_track_ids.append(track_id)
            tracks[track_id].update(
                bbox=component["bbox"],
                centroid=component["centroid"],
                last_frame_index=frame_index,
            )
            tracks[track_id].setdefault("centroids", []).append(component["centroid"])
            tracks[track_id]["samples"].append(component["samples"])
            component_refs.append((frame_index, track_id, component["mask"]))
        cooccurring_track_ids_by_frame.append(visible_track_ids)

    track_targets = {
        track_id: np.median(np.concatenate(track["samples"], axis=0), axis=0).astype(np.float32)
        for track_id, track in tracks.items()
        if track["samples"]
    }
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
    geometry_cache: dict[int, list[dict[str, object]]],
) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for mask in masks:
        geometry = geometry_cache.get(id(mask))
        if geometry is None:
            component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            geometry = [
                {
                    "mask": labels == index,
                    "bbox": [
                        int(stats[index, cv2.CC_STAT_LEFT]),
                        int(stats[index, cv2.CC_STAT_TOP]),
                        int(stats[index, cv2.CC_STAT_WIDTH]),
                        int(stats[index, cv2.CC_STAT_HEIGHT]),
                    ],
                    "centroid": (float(centroids[index][0]), float(centroids[index][1])),
                }
                for index in range(1, component_count)
                if int(stats[index, cv2.CC_STAT_AREA]) >= min_area
            ]
            geometry_cache[id(mask)] = geometry
        for item in geometry:
            confident_mask = item["mask"] & (model_chroma >= model_chroma_min)
            confident_count = int(np.count_nonzero(confident_mask))
            if confident_count < max(20, min_area // 20):
                continue
            samples = model_ab[confident_mask]
            components.append(
                {
                    "mask": item["mask"],
                    "bbox": item["bbox"],
                    "centroid": item["centroid"],
                    "samples": samples[:: max(1, len(samples) // 500)].astype(np.float32),
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
        score = _bbox_iou(component["bbox"], track.get("bbox"))
        score += max(0.0, 1.0 - _centroid_distance(component["centroid"], track.get("centroid")) / 140.0)
        if score > best_score:
            best_score = score
            best_track_id = track_id
    return best_track_id if best_score >= 0.25 else None


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
                if float(np.linalg.norm(track_targets[first_id] - track_targets[second_id])) < threshold:
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
    track_ids = sorted({track_id for edge in close_edges for track_id in edge}, key=lambda track_id: _track_x(tracks, track_id))
    for order, track_id in enumerate(track_ids):
        track_targets[track_id] = (1.0 - strength) * track_targets[track_id] + strength * palette[order % len(palette)]


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
    return intersection / max(w1 * h1 + w2 * h2 - intersection, 1e-6)


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
    mix_cache: dict[int, np.ndarray],
) -> np.ndarray:
    output = ab.copy()
    for component_mask, target_ab in targets:
        mix = mix_cache.get(id(component_mask))
        if mix is None:
            mix_mask = component_mask.astype(np.float32)
            if feather_sigma > 0.0:
                mix_mask = cv2.GaussianBlur(mix_mask, (0, 0), feather_sigma)
            mix = np.clip(mix_mask * strength, 0.0, 1.0)[:, :, None]
            mix_cache[id(component_mask)] = mix
        output = (1.0 - mix) * output + mix * target_ab
    return output


def _hex_to_lab_ab(value: str) -> np.ndarray:
    value = value.removeprefix("#")
    rgb = np.array([[[int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)]]], dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0, 1:3]


def _read_frames(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    frame_bytes = width * height * 3
    reader = open_rawvideo_reader(input_path=path)
    if reader.stdout is None or reader.stderr is None:
        raise RuntimeError("ffmpeg rawvideo reader failed to expose stdout/stderr pipes.")
    frames: list[np.ndarray] = []
    try:
        while frame_data := reader.stdout.read(frame_bytes):
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
    preset: str,
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
        preset=preset,
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
