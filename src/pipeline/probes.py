from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import yaml

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import (
    extract_clip,
    ffprobe_media,
    get_media_duration_seconds,
)
from src.pipeline.manifest import write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


TIMECODE_PATTERN = re.compile(
    r"^(?P<hours>\d{1,2}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)(?:\.(?P<millis>\d{1,3}))?$"
)


@dataclass(frozen=True)
class ProbeSpec:
    clip_id: str
    start_time: str
    end_time: str
    notes: str = ""


@dataclass(frozen=True)
class ProbeResult:
    clip_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    fps: str
    width: int
    height: int
    source_path: str
    output_path: str
    notes: str


def run_extract_probes(
    *,
    config: AppConfig,
    config_path: Path,
    movie_path: Path,
    clip_specs: list[str],
    clip_file: Path | None,
    manifest_path: Path | None,
    force: bool,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    movie_path = movie_path.expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file not found: {movie_path}")

    probe_specs = load_probe_specs(clip_specs=clip_specs, clip_file=clip_file)
    if not probe_specs:
        raise ValueError("No probe clips were provided. Use --clip or --clip-file.")

    movie_duration = get_media_duration_seconds(movie_path)
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else (paths.manifest_dir / "probe_clips.json")
    )

    print(f"Config: {config_path.resolve()}")
    print(f"Movie: {movie_path}")
    print(f"Movie duration seconds: {movie_duration:.3f}")
    print(f"Probe output dir: {paths.probe_dir}")
    print(f"Manifest path: {manifest_path}")

    results: list[ProbeResult] = []
    for spec in probe_specs:
        validate_probe_spec(spec, movie_duration_seconds=movie_duration)
        output_path = paths.probe_dir / f"{spec.clip_id}.mp4"

        if output_path.exists() and not force:
            print(f"Skipping existing probe: {output_path.name}")
        else:
            extract_clip(
                input_path=movie_path,
                output_path=output_path,
                start_time=spec.start_time,
                end_time=spec.end_time,
                video_codec=str(config.raw["video"]["output_codec"]),
                crf=int(config.raw["video"]["crf"]),
                pixel_format=str(config.raw["video"]["pixel_format"]),
            )
            print(f"Extracted probe: {output_path.name}")

        media_info = ffprobe_media(output_path)
        results.append(
            ProbeResult(
                clip_id=spec.clip_id,
                start_time=spec.start_time,
                end_time=spec.end_time,
                duration_seconds=float(media_info["duration_seconds"]),
                fps=str(media_info["fps"]),
                width=int(media_info["width"]),
                height=int(media_info["height"]),
                source_path=str(movie_path),
                output_path=str(output_path),
                notes=spec.notes,
            )
        )

    payload: dict[str, Any] = {
        "movie": str(movie_path),
        "config_path": str(config_path.resolve()),
        "clips": [asdict(result) for result in results],
    }
    write_json_manifest(manifest_path, payload)
    print(f"Probe manifest written to {manifest_path}")
    return 0


def load_probe_specs(*, clip_specs: list[str], clip_file: Path | None) -> list[ProbeSpec]:
    probe_specs: list[ProbeSpec] = []
    if clip_file is not None:
        probe_specs.extend(load_probe_specs_from_file(clip_file))
    probe_specs.extend(parse_clip_spec(item) for item in clip_specs)

    seen_ids: set[str] = set()
    for spec in probe_specs:
        if spec.clip_id in seen_ids:
            raise ValueError(f"Duplicate clip_id detected: {spec.clip_id}")
        seen_ids.add(spec.clip_id)
    return probe_specs


def load_probe_specs_from_file(path: Path) -> list[ProbeSpec]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Clip file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raw = yaml.safe_load(path.read_text())

    if not isinstance(raw, list):
        raise ValueError("Clip file must contain a list of clip objects.")

    specs: list[ProbeSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each clip file entry must be an object.")
        specs.append(
            ProbeSpec(
                clip_id=str(item["clip_id"]),
                start_time=str(item["start_time"]),
                end_time=str(item["end_time"]),
                notes=str(item.get("notes", "")),
            )
        )
    return specs


def parse_clip_spec(value: str) -> ProbeSpec:
    if "=" not in value:
        raise ValueError(
            f"Invalid clip spec '{value}'. Expected format: clip_id=HH:MM:SS-HH:MM:SS"
        )

    clip_id, remainder = value.split("=", 1)
    notes = ""
    if "|" in remainder:
        time_range, notes = remainder.split("|", 1)
    else:
        time_range = remainder

    if "-" not in time_range:
        raise ValueError(
            f"Invalid clip spec '{value}'. Expected format: clip_id=HH:MM:SS-HH:MM:SS"
        )

    start_time, end_time = time_range.split("-", 1)
    return ProbeSpec(
        clip_id=clip_id.strip(),
        start_time=start_time.strip(),
        end_time=end_time.strip(),
        notes=notes.strip(),
    )


def validate_probe_spec(spec: ProbeSpec, *, movie_duration_seconds: float) -> None:
    if not spec.clip_id:
        raise ValueError("clip_id must not be empty.")

    start_seconds = parse_timecode_to_seconds(spec.start_time)
    end_seconds = parse_timecode_to_seconds(spec.end_time)

    if end_seconds <= start_seconds:
        raise ValueError(
            f"Probe '{spec.clip_id}' has end_time <= start_time: {spec.start_time} -> {spec.end_time}"
        )
    if start_seconds >= movie_duration_seconds:
        raise ValueError(
            f"Probe '{spec.clip_id}' starts outside the movie duration: {spec.start_time}"
        )
    if end_seconds > movie_duration_seconds:
        raise ValueError(
            f"Probe '{spec.clip_id}' ends outside the movie duration: {spec.end_time}"
        )


def parse_timecode_to_seconds(value: str) -> float:
    match = TIMECODE_PATTERN.match(value)
    if not match:
        raise ValueError(
            f"Invalid timecode '{value}'. Expected HH:MM:SS or HH:MM:SS.mmm"
        )

    millis_text = match.group("millis") or "0"
    millis = int(millis_text.ljust(3, "0"))
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
        + millis / 1000.0
    )
