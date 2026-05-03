from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pipeline.manifest import load_json_manifest
from src.pipeline.segments import load_segment_manifest


def run_manifest_stats(
    *,
    segment_manifest_path: Path,
    palette_manifest_path: Path | None,
    output_path: Path | None,
) -> int:
    segment_manifest_path = segment_manifest_path.expanduser().resolve()
    if not segment_manifest_path.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest_path}")
    palette_manifest_path = palette_manifest_path.expanduser().resolve() if palette_manifest_path else None
    if palette_manifest_path is not None and not palette_manifest_path.exists():
        raise FileNotFoundError(f"Palette manifest not found: {palette_manifest_path}")

    manifest = load_segment_manifest(segment_manifest_path)
    palette = _load_palette(palette_manifest_path)
    stats = manifest_stats(manifest=manifest, palette=palette)
    text = _format_stats(stats)
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        print(f"Manifest stats written: {output_path}")
    else:
        print(text, end="")
    return 0


def manifest_stats(*, manifest: dict[str, Any], palette: dict[str, str]) -> dict[str, Any]:
    frame_count = int(manifest["frame_count"])
    tracks = manifest["tracks"]
    track_frames: dict[str, list[int]] = {track_id: [] for track_id in tracks}
    track_confidences: dict[str, list[float]] = {track_id: [] for track_id in tracks}
    empty_frames = 0
    for frame in manifest["frames"]:
        instances = frame.get("instances", [])
        if not instances:
            empty_frames += 1
        for instance in instances:
            track_id = str(instance["track_id"])
            track_frames.setdefault(track_id, []).append(int(frame["frame_index"]))
            track_confidences.setdefault(track_id, []).append(float(instance["confidence"]))

    actor_tracks = [
        track_id
        for track_id, track in tracks.items()
        if str(track.get("label")) == "actor" or "actor" in str(track.get("kind", ""))
    ]
    visible_tracks = sorted(track_id for track_id, frames in track_frames.items() if frames)
    one_frame_tracks = sorted(track_id for track_id, frames in track_frames.items() if len(frames) == 1)
    missing_palette_tracks = sorted(track_id for track_id in visible_tracks if palette and track_id not in palette)
    unused_palette_tracks = sorted(track_id for track_id in palette if track_id not in visible_tracks)

    return {
        "backend": manifest["backend"],
        "frame_count": frame_count,
        "track_count": len(tracks),
        "actor_track_count": len(actor_tracks),
        "costume_track_count": len(tracks) - len(actor_tracks),
        "empty_frame_count": empty_frames,
        "one_frame_tracks": one_frame_tracks,
        "missing_palette_tracks": missing_palette_tracks,
        "unused_palette_tracks": unused_palette_tracks,
        "tracks": {
            track_id: {
                "label": tracks.get(track_id, {}).get("label"),
                "kind": tracks.get(track_id, {}).get("kind"),
                "parent_track_id": tracks.get(track_id, {}).get("parent_track_id"),
                "frame_count": len(frames),
                "coverage": len(frames) / max(frame_count, 1),
                "span": [min(frames), max(frames)] if frames else None,
                "max_missing_span": _max_missing_span(frames),
                "mean_confidence": (
                    sum(track_confidences.get(track_id, [])) / len(track_confidences.get(track_id, []))
                    if track_confidences.get(track_id)
                    else None
                ),
            }
            for track_id, frames in sorted(track_frames.items())
        },
    }


def _load_palette(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = load_json_manifest(path, {})
    if not isinstance(payload, dict):
        raise ValueError(f"Palette manifest must be an object: {path}")
    tracks = payload.get("tracks", payload)
    if not isinstance(tracks, dict):
        raise ValueError(f"Palette manifest tracks must be an object: {path}")
    palette = {str(track_id): str(color) for track_id, color in tracks.items()}
    aliases = payload.get("aliases", {})
    if aliases:
        if not isinstance(aliases, dict):
            raise ValueError(f"Palette aliases must be an object: {path}")
        for alias_track_id, target_track_id in aliases.items():
            target_color = palette.get(str(target_track_id))
            if target_color is not None:
                palette[str(alias_track_id)] = target_color
    return palette


def _max_missing_span(frames: list[int]) -> int:
    if len(frames) <= 1:
        return 0
    sorted_frames = sorted(set(frames))
    return max((right - left - 1 for left, right in zip(sorted_frames, sorted_frames[1:], strict=False)), default=0)


def _format_stats(stats: dict[str, Any]) -> str:
    lines = [
        f"backend: {stats['backend']}",
        f"frame_count: {stats['frame_count']}",
        f"track_count: {stats['track_count']}",
        f"actor_track_count: {stats['actor_track_count']}",
        f"costume_track_count: {stats['costume_track_count']}",
        f"empty_frame_count: {stats['empty_frame_count']}",
        f"one_frame_tracks: {len(stats['one_frame_tracks'])}",
        f"missing_palette_tracks: {len(stats['missing_palette_tracks'])}",
        f"unused_palette_tracks: {len(stats['unused_palette_tracks'])}",
        "",
    ]
    if stats["one_frame_tracks"]:
        lines.append("one_frame_track_ids: " + ", ".join(stats["one_frame_tracks"]))
    if stats["missing_palette_tracks"]:
        lines.append("missing_palette_track_ids: " + ", ".join(stats["missing_palette_tracks"]))
    if stats["unused_palette_tracks"]:
        lines.append("unused_palette_track_ids: " + ", ".join(stats["unused_palette_tracks"]))
    lines.append("")
    lines.append("tracks:")
    for track_id, track in stats["tracks"].items():
        lines.append(
            f"- {track_id}: label={track['label']} parent={track['parent_track_id']} "
            f"frames={track['frame_count']} coverage={track['coverage']:.3f} "
            f"span={track['span']} max_missing={track['max_missing_span']} "
            f"confidence={_format_confidence(track['mean_confidence'])}"
        )
    return "\n".join(lines) + "\n"


def _format_confidence(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
