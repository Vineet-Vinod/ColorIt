from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest


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
    write_json_manifest(path, segment_manifest_to_dict(manifest))


def load_segment_manifest(path: Path) -> dict[str, Any]:
    payload = load_json_manifest(path, {})
    if not isinstance(payload, dict):
        raise ValueError(f"Segment manifest must be a JSON object: {path}")
    if int(payload.get("version", 0)) != SEGMENT_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported segment manifest version {payload.get('version')}; "
            f"expected {SEGMENT_MANIFEST_VERSION}: {path}"
        )
    return payload


def segment_manifest_to_dict(manifest: SegmentManifest) -> dict[str, Any]:
    payload = asdict(manifest)
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


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
