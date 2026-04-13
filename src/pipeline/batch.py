from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from src.pipeline.colorize_clip import run_colorize_clip
from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import extract_clip
from src.pipeline.manifest import write_json_manifest
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest


@dataclass(frozen=True)
class BatchSceneStatus:
    scene_id: str
    input_clip: str
    output_clip: str
    status: str
    runtime_seconds: float | None = None
    error: str | None = None


def run_colorize_batch(
    *,
    config: AppConfig,
    config_path: Path,
    movie_path: Path,
    scene_manifest_path: Path,
    resume: bool,
    limit: int | None,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    movie_path = movie_path.expanduser().resolve()
    scene_manifest_path = scene_manifest_path.expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file not found: {movie_path}")

    manifest = load_scene_manifest(scene_manifest_path)
    scenes = manifest["scenes"]
    if limit is not None:
        scenes = scenes[:limit]

    run_id = scene_manifest_path.stem
    scene_output_dir = paths.scene_dir / run_id
    colorized_output_dir = paths.colorized_dir / "scenes" / run_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    colorized_output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_scene_clips = bool(config.raw.get("runtime", {}).get("cleanup_scene_clips", True))

    batch_manifest_path = paths.manifest_dir / f"full_run_{run_id}.json"
    scene_runs_manifest_path = paths.manifest_dir / f"scene_runs_{run_id}.json"
    batch_payload: dict[str, Any]
    if resume and batch_manifest_path.exists():
        batch_payload = json.loads(batch_manifest_path.read_text())
    else:
        batch_payload = {
            "movie": str(movie_path),
            "config_path": str(config_path.resolve()),
            "scene_manifest_path": str(scene_manifest_path),
            "runs": [],
        }

    completed_outputs = {entry["output_clip"] for entry in batch_payload["runs"] if entry["status"] == "succeeded"}
    succeeded = 0
    failed = 0

    print(f"Movie: {movie_path}")
    print(f"Scene manifest: {scene_manifest_path}")
    print(f"Batch scene count: {len(scenes)}")
    print(f"Resume mode: {resume}")
    print("Loading colorizer model once for batch reuse...")
    shared_bundle = load_colorizer_bundle(config)

    for scene in scenes:
        scene_id = scene["scene_id"]
        scene_clip_path = scene_output_dir / f"{scene_id}.mp4"
        colorized_clip_path = colorized_output_dir / f"{scene_id}.mp4"

        if resume and str(colorized_clip_path) in completed_outputs and colorized_clip_path.exists():
            print(f"Skipping completed scene: {scene_id}")
            continue

        started = time.perf_counter()
        try:
            if not scene_clip_path.exists():
                extract_clip(
                    input_path=movie_path,
                    output_path=scene_clip_path,
                    start_time=scene["start_time"],
                    end_time=scene["end_time"],
                    video_codec=str(config.raw["video"]["output_codec"]),
                    crf=int(config.raw["video"]["crf"]),
                    pixel_format=str(config.raw["video"]["pixel_format"]),
                )
                print(f"Extracted scene clip: {scene_clip_path.name}")

            run_colorize_clip(
                config=config,
                config_path=config_path,
                input_path=scene_clip_path,
                output_path=colorized_clip_path,
                manifest_path=scene_runs_manifest_path,
                overwrite=True,
                model_bundle=shared_bundle,
            )
            succeeded += 1
            status = BatchSceneStatus(
                scene_id=scene_id,
                input_clip=str(scene_clip_path),
                output_clip=str(colorized_clip_path),
                status="succeeded",
                runtime_seconds=time.perf_counter() - started,
            )
            if cleanup_scene_clips and scene_clip_path.exists():
                scene_clip_path.unlink()
        except Exception as exc:
            failed += 1
            status = BatchSceneStatus(
                scene_id=scene_id,
                input_clip=str(scene_clip_path),
                output_clip=str(colorized_clip_path),
                status="failed",
                runtime_seconds=time.perf_counter() - started,
                error=str(exc),
            )
            print(f"Scene failed: {scene_id}: {exc}")

        batch_payload["runs"].append(asdict(status))
        write_json_manifest(batch_manifest_path, batch_payload)

    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Batch manifest written to {batch_manifest_path}")
    return 0
