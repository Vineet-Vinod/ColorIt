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
class ActiveTrack:
    track_id: str
    last_frame_index: int
    bbox: list[int]
    centroid: tuple[float, float]
    area: int
    missed_frames: int = 0


@dataclass(frozen=True)
class Component:
    mask: np.ndarray
    bbox: list[int]
    centroid: tuple[float, float]
    area: int


def run_track_segments(
    *,
    segment_manifest_path: Path,
    output_dir: Path,
    include_labels: list[str],
    output_label: str,
    min_area: int,
    iou_threshold: float,
    max_center_distance: float,
    max_missing_frames: int,
    split_wide_components: bool,
    preserve_source_instances: bool,
    max_component_width_ratio: float,
    min_split_valley_ratio: float,
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
    width = int(source_manifest["width"])
    height = int(source_manifest["height"])
    include_label_set = set(include_labels)
    if not include_label_set:
        include_label_set = {str(track["label"]) for track in source_manifest["tracks"].values()}

    next_track_number = 1
    active_tracks: dict[str, ActiveTrack] = {}
    output_tracks: dict[str, SegmentTrack] = {}
    output_frames: list[SegmentFrame] = []

    for frame_payload in source_manifest["frames"]:
        frame_index = int(frame_payload["frame_index"])
        frame_masks = _frame_masks(
            frame_payload=frame_payload,
            segment_manifest_path=segment_manifest_path,
            include_labels=include_label_set,
            width=width,
            height=height,
            preserve_source_instances=preserve_source_instances,
        )
        components = _components_from_masks(
            frame_masks,
            min_area=min_area,
            split_wide_components=split_wide_components,
            max_component_width=int(round(width * max_component_width_ratio)),
            min_split_valley_ratio=min_split_valley_ratio,
        )
        assignments = _assign_components_to_tracks(
            components=components,
            active_tracks=active_tracks,
            frame_index=frame_index,
            iou_threshold=iou_threshold,
            max_center_distance=max_center_distance,
            max_missing_frames=max_missing_frames,
        )

        instances: list[SegmentInstance] = []
        assigned_track_ids: set[str] = set()
        for component_index, component in enumerate(components):
            track_id = assignments.get(component_index)
            if track_id is None:
                track_id = f"{output_label}_{next_track_number:03d}"
                next_track_number += 1
                output_tracks[track_id] = SegmentTrack(
                    track_id=track_id,
                    label=output_label,
                    kind="tracked_segment",
                )
            assigned_track_ids.add(track_id)
            active_tracks[track_id] = ActiveTrack(
                track_id=track_id,
                last_frame_index=frame_index,
                bbox=component.bbox,
                centroid=component.centroid,
                area=component.area,
                missed_frames=0,
            )
            output_tracks.setdefault(
                track_id,
                SegmentTrack(track_id=track_id, label=output_label, kind="tracked_segment"),
            )

            mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{track_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(component.mask).save(mask_path)
            instances.append(
                SegmentInstance(
                    track_id=track_id,
                    label=output_label,
                    kind="tracked_segment",
                    mask_path=str(mask_path.relative_to(output_dir)),
                    bbox=component.bbox,
                    confidence=1.0,
                )
            )

        _age_unassigned_tracks(
            active_tracks=active_tracks,
            assigned_track_ids=assigned_track_ids,
            max_missing_frames=max_missing_frames,
        )
        output_frames.append(SegmentFrame(frame_index=frame_index, instances=instances))

    manifest = SegmentManifest(
        source_clip=str(source_manifest["source_clip"]),
        backend="track-segments",
        frame_count=int(source_manifest["frame_count"]),
        width=width,
        height=height,
        fps=str(source_manifest["fps"]),
        tracks=output_tracks,
        frames=output_frames,
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest)
    print(f"Tracked segment manifest written: {output_dir / 'segment_manifest.json'}")
    print(f"Track count: {len(output_tracks)}")
    return 0


def _frame_masks(
    *,
    frame_payload: dict,
    segment_manifest_path: Path,
    include_labels: set[str],
    width: int,
    height: int,
    preserve_source_instances: bool,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    combined = np.zeros((height, width), dtype=np.uint8)
    for instance in frame_payload.get("instances", []):
        if str(instance.get("label", "")) not in include_labels:
            continue
        mask_path = resolve_manifest_path(
            manifest_path=segment_manifest_path,
            relative_path=str(instance["mask_path"]),
        )
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if preserve_source_instances:
            masks.append(mask)
        else:
            combined = cv2.bitwise_or(combined, mask)
    return masks if preserve_source_instances else [combined]


def _components_from_masks(
    masks: list[np.ndarray],
    *,
    min_area: int,
    split_wide_components: bool,
    max_component_width: int,
    min_split_valley_ratio: float,
) -> list[Component]:
    components: list[Component] = []
    for mask in masks:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            component_mask[labels == component_index] = 255
            split_masks = [component_mask]
            if split_wide_components:
                split_masks = _split_wide_component(
                    component_mask,
                    min_area=min_area,
                    max_component_width=max_component_width,
                    min_split_valley_ratio=min_split_valley_ratio,
                )
            for split_mask in split_masks:
                split_area = int(np.count_nonzero(split_mask))
                if split_area < min_area:
                    continue
                bbox = mask_bbox(split_mask)
                moments = cv2.moments(split_mask, binaryImage=True)
                if moments["m00"] == 0:
                    continue
                components.append(
                    Component(
                        mask=split_mask,
                        bbox=bbox,
                        centroid=(float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])),
                        area=split_area,
                    )
                )
    components.sort(key=lambda component: component.bbox[0])
    return components


def _split_wide_component(
    mask: np.ndarray,
    *,
    min_area: int,
    max_component_width: int,
    min_split_valley_ratio: float,
) -> list[np.ndarray]:
    bbox = mask_bbox(mask)
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    if width <= max(max_component_width, 1):
        return [mask]

    cropped = mask[y1:y2, x1:x2] > 0
    projection = cropped.sum(axis=0).astype(np.float32)
    if projection.size < 3:
        return [mask]
    smooth_width = max(5, min(31, (projection.size // 12) | 1))
    kernel = np.ones(smooth_width, dtype=np.float32) / float(smooth_width)
    smoothed = np.convolve(projection, kernel, mode="same")

    margin = max(4, int(round(projection.size * 0.18)))
    if margin * 2 >= projection.size:
        return [mask]
    search = smoothed[margin:-margin]
    if search.size == 0:
        return [mask]

    split_x = int(np.argmin(search) + margin)
    valley = float(smoothed[split_x])
    average = float(smoothed.mean())
    if average <= 0.0 or valley > average * float(min_split_valley_ratio):
        return [mask]

    left = np.zeros(mask.shape, dtype=np.uint8)
    right = np.zeros(mask.shape, dtype=np.uint8)
    left[:, : x1 + split_x] = mask[:, : x1 + split_x]
    right[:, x1 + split_x :] = mask[:, x1 + split_x :]
    if np.count_nonzero(left) < min_area or np.count_nonzero(right) < min_area:
        return [mask]

    output: list[np.ndarray] = []
    for split_mask in (left, right):
        output.extend(
            _split_wide_component(
                split_mask,
                min_area=min_area,
                max_component_width=max_component_width,
                min_split_valley_ratio=min_split_valley_ratio,
            )
        )
    return output


def _assign_components_to_tracks(
    *,
    components: list[Component],
    active_tracks: dict[str, ActiveTrack],
    frame_index: int,
    iou_threshold: float,
    max_center_distance: float,
    max_missing_frames: int,
) -> dict[int, str]:
    candidates: list[tuple[float, int, str]] = []
    for component_index, component in enumerate(components):
        for track_id, track in active_tracks.items():
            if frame_index - track.last_frame_index > max_missing_frames + 1:
                continue
            iou = _bbox_iou(component.bbox, track.bbox)
            center_distance = _center_distance(component.centroid, track.centroid)
            if iou < iou_threshold and center_distance > max_center_distance:
                continue
            score = iou - (center_distance / max(max_center_distance, 1.0) * 0.05)
            candidates.append((score, component_index, track_id))

    assignments: dict[int, str] = {}
    used_tracks: set[str] = set()
    for _, component_index, track_id in sorted(candidates, reverse=True):
        if component_index in assignments or track_id in used_tracks:
            continue
        assignments[component_index] = track_id
        used_tracks.add(track_id)
    return assignments


def _age_unassigned_tracks(
    *,
    active_tracks: dict[str, ActiveTrack],
    assigned_track_ids: set[str],
    max_missing_frames: int,
) -> None:
    stale_track_ids: list[str] = []
    for track_id, track in active_tracks.items():
        if track_id in assigned_track_ids:
            continue
        track.missed_frames += 1
        if track.missed_frames > max_missing_frames:
            stale_track_ids.append(track_id)
    for track_id in stale_track_ids:
        active_tracks.pop(track_id, None)


def _bbox_iou(left: list[int], right: list[int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    return intersection / float(left_area + right_area - intersection)


def _center_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))
