from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.pipeline.segments import (
    SegmentFrame,
    SegmentInstance,
    SegmentManifest,
    SegmentTrack,
    load_segment_manifest,
    mask_bbox,
    resolve_manifest_path,
    write_segment_manifest,
)


@dataclass
class ActiveGuideTrack:
    track_id: str
    centroid: tuple[float, float]
    last_frame_index: int
    missed_frames: int = 0


def run_filter_segments(
    *,
    segment_manifest_path: Path,
    output_dir: Path,
    include_labels: list[str],
    veto_labels: list[str],
    split_guide_labels: list[str],
    output_label: str,
    min_area: int,
    guide_min_area: int,
    guide_merge_distance: float,
    track_split_guides: bool,
    guide_track_max_distance: float,
    guide_track_max_missing: int,
    close_px: int,
    erode_px: int,
    dilate_px: int,
    veto_dilate_px: int,
    overwrite: bool,
) -> int:
    segment_manifest_path = segment_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segment output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = load_segment_manifest(segment_manifest_path)
    include_label_set = set(include_labels)
    veto_label_set = set(veto_labels)
    split_guide_label_set = set(split_guide_labels)
    if not include_label_set:
        raise ValueError("At least one --include-label value is required.")

    output_tracks: dict[str, SegmentTrack] = {}

    frames: list[SegmentFrame] = []
    width = int(source_manifest["width"])
    height = int(source_manifest["height"])
    next_guide_track_number = 1
    active_guide_tracks: dict[str, ActiveGuideTrack] = {}
    for frame_payload in source_manifest["frames"]:
        frame_index = int(frame_payload["frame_index"])
        include_mask = np.zeros((height, width), dtype=np.uint8)
        veto_mask = np.zeros((height, width), dtype=np.uint8)
        guide_mask = np.zeros((height, width), dtype=np.uint8)

        for instance in frame_payload.get("instances", []):
            label = str(instance.get("label", ""))
            if label not in include_label_set and label not in veto_label_set and label not in split_guide_label_set:
                continue
            mask = _load_instance_mask(
                segment_manifest_path=segment_manifest_path,
                instance=instance,
                width=width,
                height=height,
            )
            if label in include_label_set:
                include_mask = cv2.bitwise_or(include_mask, mask)
            if label in veto_label_set:
                veto_mask = cv2.bitwise_or(veto_mask, mask)
            if label in split_guide_label_set:
                guide_mask = cv2.bitwise_or(guide_mask, mask)

        if veto_dilate_px > 0:
            veto_mask = cv2.dilate(veto_mask, _kernel(veto_dilate_px), iterations=1)
        output_mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(veto_mask))
        output_mask = _clean_mask(
            output_mask,
            close_px=close_px,
            erode_px=erode_px,
            dilate_px=dilate_px,
            min_area=min_area,
        )

        instances: list[SegmentInstance] = []
        guide_anchors = _guide_anchors(
            guide_mask=guide_mask,
            guide_min_area=guide_min_area,
            guide_merge_distance=guide_merge_distance,
        )
        if track_split_guides:
            assigned_guides, next_guide_track_number = _assign_guides_to_tracks(
                anchors=guide_anchors,
                active_tracks=active_guide_tracks,
                frame_index=frame_index,
                next_track_number=next_guide_track_number,
                max_distance=guide_track_max_distance,
                max_missing=guide_track_max_missing,
            )
            split_masks = _split_mask_by_anchors(
                mask=output_mask,
                guide_anchors=[(track_id, x, y) for track_id, (x, y) in assigned_guides],
                min_area=min_area,
            )
        else:
            split_masks = _split_mask_by_anchors(
                mask=output_mask,
                guide_anchors=[
                    (f"{index:02d}", x, y)
                    for index, (x, y) in enumerate(guide_anchors, start=1)
                ],
                min_area=min_area,
            )
        for split_id, split_mask in split_masks:
            if not np.any(split_mask):
                continue
            track_id = f"filtered_{output_label}_{split_id}"
            output_tracks.setdefault(
                track_id,
                SegmentTrack(
                    track_id=track_id,
                    label=output_label,
                    kind="filtered_segment",
                ),
            )
            mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{track_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(split_mask).save(mask_path)
            instances.append(
                SegmentInstance(
                    track_id=track_id,
                    label=output_label,
                    kind="filtered_segment",
                    mask_path=str(mask_path.relative_to(output_dir)),
                    bbox=mask_bbox(split_mask),
                    confidence=1.0,
                )
            )
        frames.append(SegmentFrame(frame_index=frame_index, instances=instances))

    manifest = SegmentManifest(
        source_clip=str(source_manifest["source_clip"]),
        backend="filter-segments",
        frame_count=int(source_manifest["frame_count"]),
        width=width,
        height=height,
        fps=str(source_manifest["fps"]),
        tracks=output_tracks,
        frames=frames,
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest)
    print(f"Filtered segment manifest written: {output_dir / 'segment_manifest.json'}")
    return 0


def _load_instance_mask(
    *,
    segment_manifest_path: Path,
    instance: dict,
    width: int,
    height: int,
) -> np.ndarray:
    mask_path = resolve_manifest_path(
        manifest_path=segment_manifest_path,
        relative_path=str(instance["mask_path"]),
    )
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _clean_mask(
    mask: np.ndarray,
    *,
    close_px: int,
    erode_px: int,
    dilate_px: int,
    min_area: int,
) -> np.ndarray:
    output = mask
    if close_px > 0:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, _kernel(close_px))
    if erode_px > 0:
        output = cv2.erode(output, _kernel(erode_px), iterations=1)
    if dilate_px > 0:
        output = cv2.dilate(output, _kernel(dilate_px), iterations=1)
    if min_area > 0:
        output = _remove_small_components(output, min_area=min_area)
    return output


def _split_mask_by_anchors(
    *,
    mask: np.ndarray,
    guide_anchors: list[tuple[str, float, float]],
    min_area: int,
) -> list[tuple[str, np.ndarray]]:
    if not np.any(mask):
        return []

    if len(guide_anchors) <= 1:
        split_id = guide_anchors[0][0] if guide_anchors else "01"
        return [(split_id, mask)]

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []

    anchor_array = np.array([(x, y) for _, x, y in guide_anchors], dtype=np.float32)
    pixel_x = xs.astype(np.float32)[:, None]
    pixel_y = ys.astype(np.float32)[:, None]
    distances = ((pixel_x - anchor_array[None, :, 0]) ** 2) + ((pixel_y - anchor_array[None, :, 1]) ** 2) * 0.35
    assignments = np.argmin(distances, axis=1)

    output: list[tuple[str, np.ndarray]] = []
    for anchor_index, (split_id, _, _) in enumerate(guide_anchors):
        split_mask = np.zeros(mask.shape, dtype=np.uint8)
        selected = assignments == anchor_index
        split_mask[ys[selected], xs[selected]] = 255
        split_mask = _remove_small_components(split_mask, min_area=min_area)
        if np.count_nonzero(split_mask) >= min_area:
            output.append((split_id, split_mask))
    return output or [(guide_anchors[0][0], mask)]


def _assign_guides_to_tracks(
    *,
    anchors: list[tuple[float, float]],
    active_tracks: dict[str, ActiveGuideTrack],
    frame_index: int,
    next_track_number: int,
    max_distance: float,
    max_missing: int,
) -> tuple[list[tuple[str, tuple[float, float]]], int]:
    candidates: list[tuple[float, int, str]] = []
    for anchor_index, anchor in enumerate(anchors):
        for track_id, track in active_tracks.items():
            if frame_index - track.last_frame_index > max_missing + 1:
                continue
            distance = float(np.hypot(anchor[0] - track.centroid[0], anchor[1] - track.centroid[1]))
            if distance <= max_distance:
                candidates.append((-distance, anchor_index, track_id))

    assignments: dict[int, str] = {}
    used_track_ids: set[str] = set()
    for _, anchor_index, track_id in sorted(candidates, reverse=True):
        if anchor_index in assignments or track_id in used_track_ids:
            continue
        assignments[anchor_index] = track_id
        used_track_ids.add(track_id)

    assigned_track_ids: set[str] = set()
    output: list[tuple[str, tuple[float, float]]] = []
    for anchor_index, anchor in enumerate(anchors):
        track_id = assignments.get(anchor_index)
        if track_id is None:
            track_id = f"guide_{next_track_number:03d}"
            next_track_number += 1
        active_tracks[track_id] = ActiveGuideTrack(
            track_id=track_id,
            centroid=anchor,
            last_frame_index=frame_index,
            missed_frames=0,
        )
        assigned_track_ids.add(track_id)
        output.append((track_id, anchor))

    stale_track_ids: list[str] = []
    for track_id, track in active_tracks.items():
        if track_id in assigned_track_ids:
            continue
        track.missed_frames += 1
        if track.missed_frames > max_missing:
            stale_track_ids.append(track_id)
    for track_id in stale_track_ids:
        active_tracks.pop(track_id, None)

    return output, next_track_number


def _guide_anchors(
    *,
    guide_mask: np.ndarray,
    guide_min_area: int,
    guide_merge_distance: float,
) -> list[tuple[float, float]]:
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(guide_mask, connectivity=8)
    components: list[tuple[int, float, float]] = []
    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area >= guide_min_area:
            components.append((area, float(centroids[component_index][0]), float(centroids[component_index][1])))

    clusters: list[tuple[float, float, int]] = []
    for area, x, y in sorted(components, reverse=True):
        best_index = None
        best_distance = float("inf")
        for cluster_index, (cx, cy, _) in enumerate(clusters):
            distance = float(np.hypot(x - cx, y - cy))
            if distance < best_distance:
                best_index = cluster_index
                best_distance = distance
        if best_index is not None and best_distance <= guide_merge_distance:
            cx, cy, total_area = clusters[best_index]
            next_area = total_area + area
            clusters[best_index] = (
                (cx * total_area + x * area) / next_area,
                (cy * total_area + y * area) / next_area,
                next_area,
            )
        else:
            clusters.append((x, y, area))

    return [(x, y) for x, y, _ in sorted(clusters, key=lambda item: item[0])]


def _remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros(mask.shape, dtype=np.uint8)
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == component_index] = 255
    return output


def _kernel(radius_px: int) -> np.ndarray:
    size = max(1, radius_px * 2 + 1)
    return np.ones((size, size), dtype=np.uint8)
