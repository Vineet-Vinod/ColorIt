from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from src.pipeline.manifest import load_json_manifest, utc_now_iso


SEGMENT_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class SegmentTrack:
    track_id: str
    label: str
    kind: str
    parent_track_id: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentInstance:
    track_id: str
    label: str
    mask_path: str
    bbox: list[int]
    confidence: float
    kind: str | None = None
    parent_track_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentFrame:
    frame_index: int
    instances: list[SegmentInstance]


@dataclass(frozen=True)
class SegmentManifest:
    source_clip: str
    backend: str
    frame_count: int
    width: int
    height: int
    fps: str
    tracks: dict[str, SegmentTrack]
    frames: list[SegmentFrame]
    created_at: str = field(default_factory=utc_now_iso)
    version: int = SEGMENT_MANIFEST_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


def write_segment_manifest(path: Path, manifest: SegmentManifest) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(segment_manifest_to_dict(manifest), separators=(",", ":")) + "\n")


def load_segment_manifest(path: Path) -> dict[str, Any]:
    payload = load_json_manifest(path, {})
    if not isinstance(payload, dict):
        raise ValueError(f"Segment manifest must be a JSON object: {path}")
    if int(payload.get("version", 0)) != SEGMENT_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported segment manifest version {payload.get('version')}; "
            f"expected {SEGMENT_MANIFEST_VERSION}: {path}"
        )
    validate_segment_manifest(payload, base_dir=path.expanduser().resolve().parent)
    return payload


def validate_segment_manifest(
    payload: dict[str, Any],
    *,
    base_dir: Path,
    require_masks: bool = True,
) -> list[str]:
    errors: list[str] = []
    _require_keys(
        payload,
        ("source_clip", "backend", "frame_count", "width", "height", "fps", "tracks", "frames"),
        errors,
        "manifest",
    )

    frame_count = _positive_int(payload.get("frame_count"), "frame_count", errors)
    width = _positive_int(payload.get("width"), "width", errors)
    height = _positive_int(payload.get("height"), "height", errors)

    tracks = payload.get("tracks", {})
    if not isinstance(tracks, dict):
        errors.append("manifest.tracks must be an object")
        tracks = {}
    for track_id, track in tracks.items():
        if not isinstance(track, dict):
            errors.append(f"tracks.{track_id} must be an object")
            continue
        if str(track.get("track_id", track_id)) != str(track_id):
            errors.append(f"tracks.{track_id}.track_id does not match object key")
        _require_keys(track, ("label", "kind"), errors, f"tracks.{track_id}")

    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        errors.append("manifest.frames must be a list")
        frames = []
    if frame_count is not None and len(frames) != frame_count:
        errors.append(f"manifest.frames length {len(frames)} does not match frame_count {frame_count}")

    seen_frames: set[int] = set()
    for frame_position, frame in enumerate(frames):
        if not isinstance(frame, dict):
            errors.append(f"frames[{frame_position}] must be an object")
            continue
        frame_index = frame.get("frame_index")
        if not isinstance(frame_index, int):
            errors.append(f"frames[{frame_position}].frame_index must be an integer")
            continue
        if frame_count is not None and not 0 <= frame_index < frame_count:
            errors.append(f"frames[{frame_position}].frame_index {frame_index} is outside frame_count")
        if frame_index in seen_frames:
            errors.append(f"duplicate frame_index: {frame_index}")
        seen_frames.add(frame_index)

        instances = frame.get("instances", [])
        if not isinstance(instances, list):
            errors.append(f"frames[{frame_position}].instances must be a list")
            continue
        for instance_position, instance in enumerate(instances):
            context = f"frames[{frame_position}].instances[{instance_position}]"
            if not isinstance(instance, dict):
                errors.append(f"{context} must be an object")
                continue
            _validate_instance(
                instance=instance,
                tracks=tracks,
                base_dir=base_dir,
                require_masks=require_masks,
                width=width,
                height=height,
                errors=errors,
                context=context,
            )

    if errors:
        raise ValueError("Invalid segment manifest:\n" + "\n".join(f"- {error}" for error in errors))
    return errors


def segment_manifest_to_dict(manifest: SegmentManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    if not payload.get("metadata"):
        payload.pop("metadata", None)
    payload["tracks"] = {
        track_id: _drop_none_values(asdict(track))
        for track_id, track in manifest.tracks.items()
    }
    payload["frames"] = [
        {
            "frame_index": frame.frame_index,
            "instances": [
                _drop_none_values(asdict(instance))
                for instance in frame.instances
            ],
        }
        for frame in manifest.frames
    ]
    return payload


def mask_bbox(mask) -> list[int]:
    rows, cols = mask.nonzero()
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0, 0, 0]
    return [int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)]


def resolve_manifest_path(*, manifest_path: Path, relative_path: str) -> Path:
    path = Path(relative_path).expanduser()
    if path.is_absolute():
        return path
    return manifest_path.expanduser().resolve().parent / path


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None and value != {} and value != []
    }


def _validate_instance(
    *,
    instance: dict[str, Any],
    tracks: dict[str, Any],
    base_dir: Path,
    require_masks: bool,
    width: int | None,
    height: int | None,
    errors: list[str],
    context: str,
) -> None:
    _require_keys(instance, ("track_id", "label", "mask_path", "bbox", "confidence"), errors, context)
    track_id = str(instance.get("track_id", ""))
    if track_id and track_id not in tracks:
        errors.append(f"{context}.track_id references unknown track: {track_id}")

    bbox = instance.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, int) for value in bbox):
        errors.append(f"{context}.bbox must be [x1, y1, x2, y2] integers")
    elif width is not None and height is not None:
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            errors.append(f"{context}.bbox {bbox} is outside {width}x{height}")

    confidence = instance.get("confidence")
    if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"{context}.confidence must be between 0 and 1")

    mask_path = instance.get("mask_path")
    if not isinstance(mask_path, str) or not mask_path:
        errors.append(f"{context}.mask_path must be a non-empty string")
    elif require_masks:
        resolved_mask_path = Path(mask_path).expanduser()
        if not resolved_mask_path.is_absolute():
            resolved_mask_path = base_dir / resolved_mask_path
        if not resolved_mask_path.exists():
            errors.append(f"{context}.mask_path does not exist: {resolved_mask_path}")


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...], errors: list[str], context: str) -> None:
    for key in keys:
        if key not in payload:
            errors.append(f"{context} is missing required key: {key}")


def _positive_int(value: Any, key: str, errors: list[str]) -> int | None:
    if not isinstance(value, int) or value <= 0:
        errors.append(f"manifest.{key} must be a positive integer")
        return None
    return value
