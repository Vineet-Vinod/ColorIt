from __future__ import annotations

from dataclasses import dataclass, field
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


DEFAULT_ACTOR_GUIDE_LABELS = ("face", "hair")
DEFAULT_CLOTHING_LABELS = ("upper_clothes", "dress", "skirt", "pants", "scarf")
DEFAULT_SKIN_LABELS = ("face", "hair", "left_arm", "right_arm", "left_leg", "right_leg")


@dataclass
class ActiveActor:
    actor_id: str
    centroid: tuple[float, float]
    last_frame_index: int
    missed_frames: int = 0


@dataclass
class TrackStats:
    actor_id: str
    garment_label: str
    frame_count: int = 0
    confidence_sum: float = 0.0
    anchors: list[tuple[float, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class ActorPartition:
    actor_id: str
    centroid: tuple[float, float]
    mask: np.ndarray | None
    missing_frames: int


def run_auto_costume_track(
    *,
    human_parser_manifest_path: Path,
    actor_manifest_path: Path | None,
    output_dir: Path,
    actor_guide_labels: list[str],
    clothing_labels: list[str],
    skin_labels: list[str],
    min_mask_area: int,
    min_confidence: float,
    min_track_frames: int,
    guide_min_area: int,
    guide_merge_distance: float,
    actor_max_distance: float,
    actor_max_missing: int,
    skin_dilate_px: int,
    actor_prior_dilate_px: int,
    close_px: int,
    erode_px: int,
    dilate_px: int,
    overwrite: bool,
) -> int:
    human_parser_manifest_path = human_parser_manifest_path.expanduser().resolve()
    if actor_manifest_path is not None:
        actor_manifest_path = actor_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not human_parser_manifest_path.exists():
        raise FileNotFoundError(f"Human parser manifest not found: {human_parser_manifest_path}")
    if actor_manifest_path is not None and not actor_manifest_path.exists():
        raise FileNotFoundError(f"Actor manifest not found: {actor_manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segment output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_guide_label_set = set(actor_guide_labels or DEFAULT_ACTOR_GUIDE_LABELS)
    clothing_label_set = set(clothing_labels or DEFAULT_CLOTHING_LABELS)
    skin_label_set = set(skin_labels or DEFAULT_SKIN_LABELS)
    manifest = load_segment_manifest(human_parser_manifest_path)
    actor_manifest = load_segment_manifest(actor_manifest_path) if actor_manifest_path is not None else None
    width = int(manifest["width"])
    height = int(manifest["height"])
    if actor_manifest is not None and (
        int(actor_manifest["width"]) != width or int(actor_manifest["height"]) != height
    ):
        raise ValueError(
            f"Actor manifest dimensions {actor_manifest['width']}x{actor_manifest['height']} do not match "
            f"human parser manifest dimensions {width}x{height}."
        )
    actor_frames_by_index = (
        {int(frame["frame_index"]): frame for frame in actor_manifest["frames"]}
        if actor_manifest is not None
        else {}
    )

    active_actors: dict[str, ActiveActor] = {}
    active_actor_partitions: dict[str, ActorPartition] = {}
    next_actor_number = 1
    output_frames: list[SegmentFrame] = []
    track_stats: dict[str, TrackStats] = {}

    for frame_payload in manifest["frames"]:
        frame_index = int(frame_payload["frame_index"])
        masks_by_label = _load_masks_by_label(
            frame_payload=frame_payload,
            manifest_path=human_parser_manifest_path,
            width=width,
            height=height,
        )
        skin_mask = _combine_masks(masks_by_label, skin_label_set, height=height, width=width)
        if skin_dilate_px > 0:
            skin_mask = cv2.dilate(skin_mask, _kernel(skin_dilate_px), iterations=1)

        actor_partitions: list[ActorPartition]
        if actor_manifest_path is not None:
            actor_partitions = _actor_partitions_from_manifest(
                frame_payload=actor_frames_by_index.get(frame_index, {"instances": []}),
                actor_manifest_path=actor_manifest_path,
                width=width,
                height=height,
            )
            actor_partitions = _carry_actor_partitions(
                current_partitions=actor_partitions,
                active_partitions=active_actor_partitions,
                actor_max_missing=actor_max_missing,
            )
        else:
            guide_mask = _combine_masks(masks_by_label, actor_guide_label_set, height=height, width=width)
            guide_anchors = _guide_anchors(
                guide_mask=guide_mask,
                guide_min_area=guide_min_area,
                guide_merge_distance=guide_merge_distance,
            )
            assigned_actors, next_actor_number = _assign_actors(
                anchors=guide_anchors,
                active_actors=active_actors,
                frame_index=frame_index,
                next_actor_number=next_actor_number,
                actor_max_distance=actor_max_distance,
                actor_max_missing=actor_max_missing,
            )
            actor_partitions = [
                ActorPartition(actor_id=actor_id, centroid=centroid, mask=None, missing_frames=missing)
                for actor_id, centroid, missing in assigned_actors
            ]

        instances: list[SegmentInstance] = []
        for clothing_label in sorted(clothing_label_set):
            clothing_mask = _combine_masks(masks_by_label, {clothing_label}, height=height, width=width)
            if not np.any(clothing_mask):
                continue
            candidate_mask = cv2.bitwise_and(clothing_mask, cv2.bitwise_not(skin_mask))
            candidate_mask = _clean_mask(
                candidate_mask,
                close_px=close_px,
                erode_px=erode_px,
                dilate_px=dilate_px,
                min_area=min_mask_area,
            )
            for actor_id, actor_mask, actor_missing in _split_by_actors(
                mask=candidate_mask,
                actors=actor_partitions,
                min_area=min_mask_area,
                actor_prior_dilate_px=actor_prior_dilate_px,
            ):
                area = int(np.count_nonzero(actor_mask))
                confidence = _candidate_confidence(
                    area=area,
                    frame_area=width * height,
                    actor_missing=actor_missing,
                    actor_max_missing=actor_max_missing,
                )
                if confidence < min_confidence:
                    continue

                track_id = f"{actor_id}_{_slug(clothing_label)}"
                mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{track_id}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(actor_mask).save(mask_path)
                instances.append(
                    SegmentInstance(
                        track_id=track_id,
                        label=clothing_label,
                        kind="auto_costume_track",
                        parent_track_id=actor_id,
                        mask_path=str(mask_path.relative_to(output_dir)),
                        bbox=mask_bbox(actor_mask),
                        confidence=confidence,
                        metadata={
                            "actor_id": actor_id,
                            "garment_label": clothing_label,
                            "actor_missing_frames": actor_missing,
                            "proposal_source": "human_parser",
                        },
                    )
                )
                stats = track_stats.setdefault(
                    track_id,
                    TrackStats(actor_id=actor_id, garment_label=clothing_label),
                )
                stats.frame_count += 1
                stats.confidence_sum += confidence
                stats.anchors.append((confidence, area, frame_index))
        output_frames.append(SegmentFrame(frame_index=frame_index, instances=instances))

    if min_track_frames > 1:
        output_frames, track_stats = _filter_short_tracks(
            output_frames=output_frames,
            track_stats=track_stats,
            min_track_frames=min_track_frames,
        )
    output_tracks = _build_tracks(track_stats)
    manifest_out = SegmentManifest(
        source_clip=str(manifest["source_clip"]),
        backend="auto-costume-track",
        frame_count=int(manifest["frame_count"]),
        width=width,
        height=height,
        fps=str(manifest["fps"]),
        tracks=output_tracks,
        frames=output_frames,
        metadata={
            "actor_manifest": str(actor_manifest_path) if actor_manifest_path is not None else None,
            "actor_guide_labels": sorted(actor_guide_label_set),
            "clothing_labels": sorted(clothing_label_set),
            "skin_labels": sorted(skin_label_set),
            "actor_prior_dilate_px": actor_prior_dilate_px,
            "min_track_frames": min_track_frames,
            "strategy": "actor_guided_human_parser_candidates",
        },
    )
    write_segment_manifest(output_dir / "segment_manifest.json", manifest_out)
    print(f"Auto costume track manifest written: {output_dir / 'segment_manifest.json'}")
    print(f"Track count: {len(output_tracks)}")
    return 0


def _load_masks_by_label(
    *,
    frame_payload: dict,
    manifest_path: Path,
    width: int,
    height: int,
) -> dict[str, list[np.ndarray]]:
    masks_by_label: dict[str, list[np.ndarray]] = {}
    for instance in frame_payload.get("instances", []):
        label = str(instance.get("label", ""))
        mask_path = resolve_manifest_path(
            manifest_path=manifest_path,
            relative_path=str(instance["mask_path"]),
        )
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks_by_label.setdefault(label, []).append(np.where(mask > 0, 255, 0).astype(np.uint8))
    return masks_by_label


def _combine_masks(
    masks_by_label: dict[str, list[np.ndarray]],
    labels: set[str],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint8)
    for label in labels:
        for mask in masks_by_label.get(label, []):
            output = cv2.bitwise_or(output, mask)
    return output


def _actor_partitions_from_manifest(
    *,
    frame_payload: dict,
    actor_manifest_path: Path,
    width: int,
    height: int,
) -> list[ActorPartition]:
    partitions: list[ActorPartition] = []
    for instance in frame_payload.get("instances", []):
        actor_id = str(instance.get("track_id", ""))
        if not actor_id:
            continue
        mask_path = resolve_manifest_path(
            manifest_path=actor_manifest_path,
            relative_path=str(instance["mask_path"]),
        )
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        moments = cv2.moments(mask, binaryImage=True)
        bbox = mask_bbox(mask)
        if moments["m00"] == 0:
            centroid = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        else:
            centroid = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
        partitions.append(ActorPartition(actor_id=actor_id, centroid=centroid, mask=mask, missing_frames=0))
    return sorted(partitions, key=lambda partition: partition.centroid[0])


def _carry_actor_partitions(
    *,
    current_partitions: list[ActorPartition],
    active_partitions: dict[str, ActorPartition],
    actor_max_missing: int,
) -> list[ActorPartition]:
    current_ids = {partition.actor_id for partition in current_partitions}
    for partition in current_partitions:
        active_partitions[partition.actor_id] = partition

    stale_actor_ids: list[str] = []
    carried_partitions = list(current_partitions)
    for actor_id, partition in list(active_partitions.items()):
        if actor_id in current_ids:
            continue
        missing_frames = partition.missing_frames + 1
        if missing_frames > actor_max_missing:
            stale_actor_ids.append(actor_id)
            continue
        carried_partition = ActorPartition(
            actor_id=actor_id,
            centroid=partition.centroid,
            mask=partition.mask,
            missing_frames=missing_frames,
        )
        active_partitions[actor_id] = carried_partition
        carried_partitions.append(carried_partition)

    for actor_id in stale_actor_ids:
        active_partitions.pop(actor_id, None)

    return sorted(carried_partitions, key=lambda partition: partition.centroid[0])


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
        match_index = None
        match_distance = float("inf")
        for cluster_index, (cx, cy, _) in enumerate(clusters):
            distance = float(np.hypot(x - cx, y - cy))
            if distance < match_distance:
                match_distance = distance
                match_index = cluster_index
        if match_index is not None and match_distance <= guide_merge_distance:
            cx, cy, total_area = clusters[match_index]
            next_area = total_area + area
            clusters[match_index] = (
                (cx * total_area + x * area) / next_area,
                (cy * total_area + y * area) / next_area,
                next_area,
            )
        else:
            clusters.append((x, y, area))
    return [(x, y) for x, y, _ in sorted(clusters, key=lambda item: item[0])]


def _assign_actors(
    *,
    anchors: list[tuple[float, float]],
    active_actors: dict[str, ActiveActor],
    frame_index: int,
    next_actor_number: int,
    actor_max_distance: float,
    actor_max_missing: int,
) -> tuple[list[tuple[str, tuple[float, float], int]], int]:
    candidates: list[tuple[float, int, str]] = []
    for anchor_index, anchor in enumerate(anchors):
        for actor_id, actor in active_actors.items():
            if frame_index - actor.last_frame_index > actor_max_missing + 1:
                continue
            distance = float(np.hypot(anchor[0] - actor.centroid[0], anchor[1] - actor.centroid[1]))
            if distance <= actor_max_distance:
                candidates.append((-distance, anchor_index, actor_id))

    assignments: dict[int, str] = {}
    used_actor_ids: set[str] = set()
    for _, anchor_index, actor_id in sorted(candidates, reverse=True):
        if anchor_index in assignments or actor_id in used_actor_ids:
            continue
        assignments[anchor_index] = actor_id
        used_actor_ids.add(actor_id)

    assigned_actor_ids: set[str] = set()
    for anchor_index, anchor in enumerate(anchors):
        actor_id = assignments.get(anchor_index)
        if actor_id is None:
            actor_id = f"actor_{next_actor_number:03d}"
            next_actor_number += 1
        active_actors[actor_id] = ActiveActor(
            actor_id=actor_id,
            centroid=anchor,
            last_frame_index=frame_index,
            missed_frames=0,
        )
        assigned_actor_ids.add(actor_id)

    stale_actor_ids: list[str] = []
    for actor_id, actor in active_actors.items():
        if actor_id in assigned_actor_ids:
            continue
        actor.missed_frames += 1
        if actor.missed_frames > actor_max_missing:
            stale_actor_ids.append(actor_id)
    for actor_id in stale_actor_ids:
        active_actors.pop(actor_id, None)

    return [
        (actor_id, actor.centroid, actor.missed_frames)
        for actor_id, actor in sorted(active_actors.items(), key=lambda item: item[1].centroid[0])
    ], next_actor_number


def _split_by_actors(
    *,
    mask: np.ndarray,
    actors: list[ActorPartition],
    min_area: int,
    actor_prior_dilate_px: int,
) -> list[tuple[str, np.ndarray, int]]:
    if not np.any(mask):
        return []
    if len(actors) <= 1:
        if not actors:
            return [("actor_001", mask, 0)]
        actor = actors[0]
        gate_mask = _actor_gate_mask(actor.mask, actor_prior_dilate_px) if actor.mask is not None else None
        actor_mask = cv2.bitwise_and(mask, gate_mask) if gate_mask is not None else mask
        actor_mask = _remove_small_components(actor_mask, min_area=min_area)
        if np.count_nonzero(actor_mask) < min_area:
            return []
        return [(actor.actor_id, actor_mask, actor.missing_frames)]

    if all(actor.mask is not None for actor in actors):
        return _split_by_actor_mask_priors(
            mask=mask,
            actors=actors,
            min_area=min_area,
            actor_prior_dilate_px=actor_prior_dilate_px,
        )

    ys, xs = np.nonzero(mask)
    anchor_array = np.array([actor.centroid for actor in actors], dtype=np.float32)
    distances = ((xs.astype(np.float32)[:, None] - anchor_array[None, :, 0]) ** 2) + (
        (ys.astype(np.float32)[:, None] - anchor_array[None, :, 1]) ** 2
    ) * 0.35
    assignments = np.argmin(distances, axis=1)

    output: list[tuple[str, np.ndarray, int]] = []
    for actor_index, actor in enumerate(actors):
        actor_mask = np.zeros(mask.shape, dtype=np.uint8)
        selected = assignments == actor_index
        actor_mask[ys[selected], xs[selected]] = 255
        actor_mask = _remove_small_components(actor_mask, min_area=min_area)
        if np.count_nonzero(actor_mask) >= min_area:
            output.append((actor.actor_id, actor_mask, actor.missing_frames))
    return output


def _split_by_actor_mask_priors(
    *,
    mask: np.ndarray,
    actors: list[ActorPartition],
    min_area: int,
    actor_prior_dilate_px: int,
) -> list[tuple[str, np.ndarray, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []

    gate_values: list[np.ndarray] = []
    actor_centroids = np.array([actor.centroid for actor in actors], dtype=np.float32)
    for actor in actors:
        assert actor.mask is not None
        gate_mask = _actor_gate_mask(actor.mask, actor_prior_dilate_px)
        gate_values.append(gate_mask[ys, xs] > 0)
    gate_array = np.stack(gate_values, axis=1)
    inside_any_gate = np.any(gate_array, axis=1)
    if not np.any(inside_any_gate):
        return []

    distances = ((xs.astype(np.float32)[:, None] - actor_centroids[None, :, 0]) ** 2) + (
        (ys.astype(np.float32)[:, None] - actor_centroids[None, :, 1]) ** 2
    ) * 0.35
    distances[~gate_array] = np.inf
    assignments = np.argmin(distances, axis=1)

    output: list[tuple[str, np.ndarray, int]] = []
    for actor_index, actor in enumerate(actors):
        actor_mask = np.zeros(mask.shape, dtype=np.uint8)
        selected = inside_any_gate & (assignments == actor_index)
        actor_mask[ys[selected], xs[selected]] = 255
        actor_mask = _remove_small_components(actor_mask, min_area=min_area)
        if np.count_nonzero(actor_mask) >= min_area:
            output.append((actor.actor_id, actor_mask, actor.missing_frames))
    return output


def _actor_gate_mask(actor_mask: np.ndarray, actor_prior_dilate_px: int) -> np.ndarray:
    if actor_prior_dilate_px <= 0:
        return actor_mask
    return cv2.dilate(actor_mask, _kernel(actor_prior_dilate_px), iterations=1)


def _candidate_confidence(
    *,
    area: int,
    frame_area: int,
    actor_missing: int,
    actor_max_missing: int,
) -> float:
    area_fraction = area / max(frame_area, 1)
    area_score = float(np.clip(area_fraction / 0.018, 0.15, 1.0))
    missing_penalty = 1.0 - min(actor_missing / max(actor_max_missing + 1, 1), 0.85)
    return float(np.clip(area_score * missing_penalty, 0.0, 1.0))


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


def _remove_small_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros(mask.shape, dtype=np.uint8)
    for component_index in range(1, component_count):
        if int(stats[component_index, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == component_index] = 255
    return output


def _build_tracks(track_stats: dict[str, TrackStats]) -> dict[str, SegmentTrack]:
    tracks: dict[str, SegmentTrack] = {}
    for stats in track_stats.values():
        tracks.setdefault(
            stats.actor_id,
            SegmentTrack(
                track_id=stats.actor_id,
                label="actor",
                kind="actor_track",
            ),
        )
    for track_id, stats in sorted(track_stats.items()):
        best_anchors = sorted(stats.anchors, reverse=True)[:5]
        tracks[track_id] = SegmentTrack(
            track_id=track_id,
            label=stats.garment_label,
            kind="auto_costume_track",
            parent_track_id=stats.actor_id,
            confidence=stats.confidence_sum / max(stats.frame_count, 1),
            metadata={
                "actor_id": stats.actor_id,
                "garment_label": stats.garment_label,
                "anchor_frame_indices": [frame_index for _, _, frame_index in best_anchors],
                "frame_count": stats.frame_count,
                "identity_scope": "actor_parent",
            },
        )
    return tracks


def _filter_short_tracks(
    *,
    output_frames: list[SegmentFrame],
    track_stats: dict[str, TrackStats],
    min_track_frames: int,
) -> tuple[list[SegmentFrame], dict[str, TrackStats]]:
    kept_track_ids = {
        track_id
        for track_id, stats in track_stats.items()
        if stats.frame_count >= min_track_frames
    }
    filtered_frames = [
        SegmentFrame(
            frame_index=frame.frame_index,
            instances=[
                instance
                for instance in frame.instances
                if instance.track_id in kept_track_ids
            ],
        )
        for frame in output_frames
    ]
    filtered_stats = {
        track_id: stats
        for track_id, stats in track_stats.items()
        if track_id in kept_track_ids
    }
    return filtered_frames, filtered_stats


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower()).strip("_")


def _kernel(radius_px: int) -> np.ndarray:
    size = max(1, radius_px * 2 + 1)
    return np.ones((size, size), dtype=np.uint8)
