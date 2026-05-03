from __future__ import annotations

from collections import defaultdict
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


@dataclass(frozen=True)
class SourceInstance:
    source_track_id: str
    frame_index: int
    mask_path: str
    bbox: list[int]
    confidence: float
    centroid: tuple[float, float]
    area: int


@dataclass
class SourceTrack:
    source_track_id: str
    instances: list[SourceInstance] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.instances[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.instances[-1].frame_index

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([instance.confidence for instance in self.instances]))


@dataclass
class StableActor:
    actor_id: str
    source_track_ids: list[str] = field(default_factory=list)
    instances: list[SourceInstance] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return min(instance.frame_index for instance in self.instances)

    @property
    def last_frame(self) -> int:
        return max(instance.frame_index for instance in self.instances)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([instance.confidence for instance in self.instances]))


def run_stitch_actors(
    *,
    actor_manifest_path: Path,
    output_dir: Path,
    min_source_frames: int,
    max_gap_frames: int,
    max_centroid_distance: float,
    min_iou: float,
    allow_overlap_frames: int,
    overwrite: bool,
) -> int:
    actor_manifest_path = actor_manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not actor_manifest_path.exists():
        raise FileNotFoundError(f"Actor manifest not found: {actor_manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Segment output dir is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_segment_manifest(actor_manifest_path)
    width = int(manifest["width"])
    height = int(manifest["height"])
    source_tracks = _source_tracks_from_manifest(manifest)
    stable_actors = _stitch_source_tracks(
        source_tracks=source_tracks,
        min_source_frames=min_source_frames,
        max_gap_frames=max_gap_frames,
        max_centroid_distance=max_centroid_distance,
        min_iou=min_iou,
        allow_overlap_frames=allow_overlap_frames,
    )

    frames = _write_stitched_masks(
        stable_actors=stable_actors,
        source_manifest_path=actor_manifest_path,
        output_dir=output_dir,
        frame_count=int(manifest["frame_count"]),
        width=width,
        height=height,
    )
    tracks = {
        actor.actor_id: SegmentTrack(
            track_id=actor.actor_id,
            label="actor",
            kind="stitched_actor",
            confidence=actor.mean_confidence,
            metadata={
                "source_track_ids": actor.source_track_ids,
                "frame_count": len({instance.frame_index for instance in actor.instances}),
                "first_frame": actor.first_frame,
                "last_frame": actor.last_frame,
                "missing_spans": _missing_spans(
                    sorted({instance.frame_index for instance in actor.instances}),
                    first_frame=actor.first_frame,
                    last_frame=actor.last_frame,
                ),
            },
        )
        for actor in stable_actors
    }
    stitched_manifest = SegmentManifest(
        source_clip=str(manifest["source_clip"]),
        backend="stitch-actors",
        frame_count=int(manifest["frame_count"]),
        width=width,
        height=height,
        fps=str(manifest["fps"]),
        tracks=tracks,
        frames=frames,
        metadata={
            "source_actor_manifest": str(actor_manifest_path),
            "min_source_frames": min_source_frames,
            "max_gap_frames": max_gap_frames,
            "max_centroid_distance": max_centroid_distance,
            "min_iou": min_iou,
            "allow_overlap_frames": allow_overlap_frames,
        },
    )
    write_segment_manifest(output_dir / "segment_manifest.json", stitched_manifest)
    _write_summary(output_dir / "actor_stitch_summary.txt", stable_actors=stable_actors, source_tracks=source_tracks)
    print(f"Stitched actor manifest written: {output_dir / 'segment_manifest.json'}")
    print(f"Stable actor count: {len(stable_actors)}")
    return 0


def _source_tracks_from_manifest(manifest: dict) -> list[SourceTrack]:
    tracks_by_id: dict[str, SourceTrack] = {}
    for frame in manifest["frames"]:
        frame_index = int(frame["frame_index"])
        for instance in frame.get("instances", []):
            source_track_id = str(instance["track_id"])
            mask_bbox_value = list(instance["bbox"])
            centroid = _bbox_center(mask_bbox_value)
            area = max(0, mask_bbox_value[2] - mask_bbox_value[0]) * max(0, mask_bbox_value[3] - mask_bbox_value[1])
            tracks_by_id.setdefault(source_track_id, SourceTrack(source_track_id=source_track_id)).instances.append(
                SourceInstance(
                    source_track_id=source_track_id,
                    frame_index=frame_index,
                    mask_path=str(instance["mask_path"]),
                    bbox=mask_bbox_value,
                    confidence=float(instance["confidence"]),
                    centroid=centroid,
                    area=area,
                )
            )
    for track in tracks_by_id.values():
        track.instances.sort(key=lambda instance: instance.frame_index)
    return sorted(tracks_by_id.values(), key=lambda track: (track.first_frame, track.instances[0].centroid[0]))


def _stitch_source_tracks(
    *,
    source_tracks: list[SourceTrack],
    min_source_frames: int,
    max_gap_frames: int,
    max_centroid_distance: float,
    min_iou: float,
    allow_overlap_frames: int,
) -> list[StableActor]:
    stable_actors: list[StableActor] = []
    next_actor_number = 1
    for source_track in source_tracks:
        best_actor: StableActor | None = None
        best_score = -1.0
        for actor in stable_actors:
            score = _stitch_score(
                actor=actor,
                source_track=source_track,
                max_gap_frames=max_gap_frames,
                max_centroid_distance=max_centroid_distance,
                min_iou=min_iou,
                allow_overlap_frames=allow_overlap_frames,
            )
            if score > best_score:
                best_score = score
                best_actor = actor
        if best_actor is None or best_score < 0.0:
            actor_id = f"actor_{next_actor_number:03d}"
            next_actor_number += 1
            best_actor = StableActor(actor_id=actor_id)
            stable_actors.append(best_actor)
        if len(source_track.instances) >= min_source_frames or not best_actor.source_track_ids:
            best_actor.source_track_ids.append(source_track.source_track_id)
            best_actor.instances.extend(source_track.instances)

    for actor in stable_actors:
        actor.instances.sort(key=lambda instance: (instance.frame_index, instance.confidence), reverse=False)
        actor.source_track_ids = list(dict.fromkeys(actor.source_track_ids))
    return sorted(stable_actors, key=lambda actor: actor.first_frame)


def _stitch_score(
    *,
    actor: StableActor,
    source_track: SourceTrack,
    max_gap_frames: int,
    max_centroid_distance: float,
    min_iou: float,
    allow_overlap_frames: int,
) -> float:
    actor_frames = {instance.frame_index for instance in actor.instances}
    source_frames = {instance.frame_index for instance in source_track.instances}
    overlap = len(actor_frames & source_frames)
    if overlap > allow_overlap_frames:
        return -1.0

    if source_track.first_frame >= actor.last_frame:
        anchor_left = _nearest_instance(actor.instances, source_track.first_frame, before=True)
        anchor_right = source_track.instances[0]
        gap = source_track.first_frame - anchor_left.frame_index
    elif actor.first_frame >= source_track.last_frame:
        anchor_left = source_track.instances[-1]
        anchor_right = _nearest_instance(actor.instances, source_track.last_frame, before=False)
        gap = actor.first_frame - source_track.last_frame
    else:
        anchor_left = _nearest_nonoverlap_instance(actor.instances, source_track.instances)
        anchor_right = _nearest_nonoverlap_instance(source_track.instances, actor.instances)
        gap = 0

    if gap > max_gap_frames:
        return -1.0
    distance = float(np.hypot(anchor_left.centroid[0] - anchor_right.centroid[0], anchor_left.centroid[1] - anchor_right.centroid[1]))
    iou = _bbox_iou(anchor_left.bbox, anchor_right.bbox)
    if distance > max_centroid_distance and iou < min_iou:
        return -1.0
    distance_score = 1.0 - min(distance / max(max_centroid_distance, 1.0), 1.0)
    gap_score = 1.0 - min(gap / max(max_gap_frames + 1, 1), 1.0)
    return 0.55 * distance_score + 0.30 * iou + 0.15 * gap_score


def _write_stitched_masks(
    *,
    stable_actors: list[StableActor],
    source_manifest_path: Path,
    output_dir: Path,
    frame_count: int,
    width: int,
    height: int,
) -> list[SegmentFrame]:
    frame_instances: dict[int, list[SegmentInstance]] = defaultdict(list)
    for actor in stable_actors:
        by_frame: dict[int, list[SourceInstance]] = defaultdict(list)
        for instance in actor.instances:
            by_frame[instance.frame_index].append(instance)
        for frame_index, instances in by_frame.items():
            mask = np.zeros((height, width), dtype=np.uint8)
            confidence_values: list[float] = []
            source_track_ids: list[str] = []
            for instance in instances:
                source_mask = _load_mask(
                    manifest_path=source_manifest_path,
                    relative_path=instance.mask_path,
                    width=width,
                    height=height,
                )
                mask = cv2.bitwise_or(mask, source_mask)
                confidence_values.append(instance.confidence)
                source_track_ids.append(instance.source_track_id)
            if not np.any(mask):
                continue
            mask_path = output_dir / "masks" / f"frame_{frame_index:06d}_{actor.actor_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mask_path)
            frame_instances[frame_index].append(
                SegmentInstance(
                    track_id=actor.actor_id,
                    label="actor",
                    kind="stitched_actor",
                    mask_path=str(mask_path.relative_to(output_dir)),
                    bbox=mask_bbox(mask),
                    confidence=float(np.mean(confidence_values)),
                    metadata={"source_track_ids": sorted(set(source_track_ids))},
                )
            )
    return [
        SegmentFrame(
            frame_index=frame_index,
            instances=sorted(frame_instances.get(frame_index, []), key=lambda instance: instance.bbox[0]),
        )
        for frame_index in range(frame_count)
    ]


def _load_mask(*, manifest_path: Path, relative_path: str, width: int, height: int) -> np.ndarray:
    mask_path = resolve_manifest_path(manifest_path=manifest_path, relative_path=relative_path)
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _write_summary(path: Path, *, stable_actors: list[StableActor], source_tracks: list[SourceTrack]) -> None:
    lines = [
        f"source_tracks: {len(source_tracks)}",
        f"stable_actors: {len(stable_actors)}",
        "",
    ]
    for actor in stable_actors:
        frames = sorted({instance.frame_index for instance in actor.instances})
        lines.append(
            f"{actor.actor_id}: frames={len(frames)} span={actor.first_frame}-{actor.last_frame} "
            f"confidence={actor.mean_confidence:.3f} sources={','.join(actor.source_track_ids)}"
        )
        missing = _missing_spans(frames, first_frame=actor.first_frame, last_frame=actor.last_frame)
        if missing:
            lines.append(f"  missing_spans={missing}")
    path.write_text("\n".join(lines) + "\n")


def _missing_spans(frames: list[int], *, first_frame: int, last_frame: int) -> list[list[int]]:
    frame_set = set(frames)
    spans: list[list[int]] = []
    start: int | None = None
    previous: int | None = None
    for frame_index in range(first_frame, last_frame + 1):
        if frame_index in frame_set:
            if start is not None and previous is not None:
                spans.append([start, previous])
                start = None
            continue
        if start is None:
            start = frame_index
        previous = frame_index
    if start is not None and previous is not None:
        spans.append([start, previous])
    return spans


def _nearest_instance(instances: list[SourceInstance], frame_index: int, *, before: bool) -> SourceInstance:
    if before:
        candidates = [instance for instance in instances if instance.frame_index <= frame_index]
        return candidates[-1] if candidates else instances[0]
    candidates = [instance for instance in instances if instance.frame_index >= frame_index]
    return candidates[0] if candidates else instances[-1]


def _nearest_nonoverlap_instance(left: list[SourceInstance], right: list[SourceInstance]) -> SourceInstance:
    right_frames = {instance.frame_index for instance in right}
    candidates = [instance for instance in left if instance.frame_index not in right_frames]
    return candidates[-1] if candidates else left[-1]


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


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
