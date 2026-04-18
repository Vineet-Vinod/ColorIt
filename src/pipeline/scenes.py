from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import detect_scene_change_times, get_media_duration_seconds
from src.pipeline.manifest import write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


@dataclass(frozen=True)
class SceneUnit:
    scene_id: str
    source_file: str
    start_time: str
    end_time: str
    duration_seconds: float
    preset: str
    scene_index: int
    parent_scene_id: str | None = None
    overlap_seconds: float = 0.0


def run_detect_scenes(
    *,
    config: AppConfig,
    config_path: Path,
    movie_path: Path,
    output_path: Path | None,
    threshold: float,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    movie_path = movie_path.expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file not found: {movie_path}")

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else paths.manifest_dir / "scenes.json"
    )

    movie_duration = get_media_duration_seconds(movie_path)
    min_scene_seconds = float(config.raw["scenes"]["min_scene_seconds"])
    max_scene_seconds = float(config.raw["scenes"]["max_scene_seconds"])
    overlap_seconds = float(config.raw["scenes"]["overlap_seconds"])

    print(f"Config: {config_path.resolve()}")
    print(f"Movie: {movie_path}")
    print(f"Scene threshold: {threshold}")
    print(f"Movie duration seconds: {movie_duration:.3f}")

    raw_boundaries = detect_scene_boundaries(
        movie_path=movie_path,
        movie_duration_seconds=movie_duration,
        threshold=threshold,
        min_scene_seconds=min_scene_seconds,
    )
    scene_units = split_scenes(
        boundaries=raw_boundaries,
        source_file=str(movie_path),
        min_scene_seconds=min_scene_seconds,
        max_scene_seconds=max_scene_seconds,
        overlap_seconds=overlap_seconds,
        preset=str(config_path.name),
    )

    payload: dict[str, Any] = {
        "movie": str(movie_path),
        "config_path": str(config_path.resolve()),
        "threshold": threshold,
        "scene_settings": {
            "min_scene_seconds": min_scene_seconds,
            "max_scene_seconds": max_scene_seconds,
            "overlap_seconds": overlap_seconds,
        },
        "scene_count": len(scene_units),
        "scenes": [asdict(scene) for scene in scene_units],
    }
    write_json_manifest(output_path, payload)
    print(f"Scene manifest written to {output_path}")
    print(f"Scene units: {len(scene_units)}")
    return 0


def detect_scene_boundaries(
    *,
    movie_path: Path,
    movie_duration_seconds: float,
    threshold: float,
    min_scene_seconds: float,
) -> list[tuple[float, float]]:
    change_times = detect_scene_change_times(movie_path=movie_path, threshold=threshold)
    boundaries = [0.0]
    for time_seconds in change_times:
        if time_seconds <= 0.0 or time_seconds >= movie_duration_seconds:
            continue
        if time_seconds - boundaries[-1] < min_scene_seconds:
            continue
        boundaries.append(time_seconds)

    if movie_duration_seconds - boundaries[-1] < min_scene_seconds and len(boundaries) > 1:
        boundaries[-1] = movie_duration_seconds
    else:
        boundaries.append(movie_duration_seconds)

    pairs: list[tuple[float, float]] = []
    for index in range(len(boundaries) - 1):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end - start >= min_scene_seconds:
            pairs.append((start, end))
    return pairs


def split_scenes(
    *,
    boundaries: list[tuple[float, float]],
    source_file: str,
    min_scene_seconds: float,
    max_scene_seconds: float,
    overlap_seconds: float,
    preset: str,
) -> list[SceneUnit]:
    units: list[SceneUnit] = []
    scene_counter = 1
    for base_index, (start, end) in enumerate(boundaries, start=1):
        duration = end - start
        if duration <= max_scene_seconds:
            units.append(
                SceneUnit(
                    scene_id=f"scene_{scene_counter:04d}",
                    source_file=source_file,
                    start_time=seconds_to_timecode(start),
                    end_time=seconds_to_timecode(end),
                    duration_seconds=round(duration, 3),
                    preset=preset,
                    scene_index=scene_counter,
                    parent_scene_id=None,
                    overlap_seconds=0.0,
                )
            )
            scene_counter += 1
            continue

        part_index = 0
        cursor = start
        parent_scene_id = f"scene_{base_index:04d}"
        step = max_scene_seconds - overlap_seconds
        if step <= 0:
            raise ValueError("max_scene_seconds must be greater than overlap_seconds.")

        while cursor < end:
            part_start = cursor
            part_end = min(end, cursor + max_scene_seconds)
            remaining_after_part = end - part_end
            if 0.0 < remaining_after_part < min_scene_seconds:
                part_end = end
            units.append(
                SceneUnit(
                    scene_id=f"{parent_scene_id}_part_{part_index:02d}",
                    source_file=source_file,
                    start_time=seconds_to_timecode(part_start),
                    end_time=seconds_to_timecode(part_end),
                    duration_seconds=round(part_end - part_start, 3),
                    preset=preset,
                    scene_index=scene_counter,
                    parent_scene_id=parent_scene_id,
                    overlap_seconds=overlap_seconds if part_index > 0 else 0.0,
                )
            )
            scene_counter += 1
            part_index += 1
            if part_end >= end:
                break
            cursor += step

    return units


def load_scene_manifest(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scene manifest not found: {path}")
    return json.loads(path.read_text())


def scene_manifest_matches(
    manifest: dict[str, Any],
    *,
    movie_path: Path,
    threshold: float,
    min_scene_seconds: float,
    max_scene_seconds: float,
    overlap_seconds: float,
) -> bool:
    manifest_movie = manifest.get("movie")
    if not manifest_movie:
        return False

    try:
        manifest_path = Path(str(manifest_movie)).expanduser().resolve()
    except Exception:
        return False

    if manifest_path != movie_path.expanduser().resolve():
        return False

    manifest_threshold = manifest.get("threshold")
    if manifest_threshold is None or abs(float(manifest_threshold) - threshold) > 1e-9:
        return False

    scene_settings = manifest.get("scene_settings")
    if not isinstance(scene_settings, dict):
        return False

    return (
        abs(float(scene_settings.get("min_scene_seconds", -1.0)) - min_scene_seconds) <= 1e-9
        and abs(float(scene_settings.get("max_scene_seconds", -1.0)) - max_scene_seconds) <= 1e-9
        and abs(float(scene_settings.get("overlap_seconds", -1.0)) - overlap_seconds) <= 1e-9
    )


def seconds_to_timecode(value: float) -> str:
    total_millis = int(round(value * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
