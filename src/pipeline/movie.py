from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import shutil
from typing import Any

from src.pipeline.assemble import run_assemble_final
from src.pipeline.batch import run_colorize_batch
from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import compress_video
from src.pipeline.manifest import load_json_manifest, utc_now_iso, write_json_manifest
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import load_scene_manifest, run_detect_scenes, scene_manifest_matches


def run_colorize_movie(
    *,
    config: AppConfig,
    config_path: Path,
    movie_path: Path,
    output_path: Path | None,
    scene_threshold: float | None,
    keep_intermediates: bool,
    resume: bool,
    limit: int | None,
    overwrite: bool,
) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    movie_path = movie_path.expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file not found: {movie_path}")

    output_path = _resolve_output_path(movie_path=movie_path, output_path=output_path, limit=limit)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    threshold = float(scene_threshold if scene_threshold is not None else config.raw["scenes"].get("threshold", 0.60))
    run_id = _build_run_id(movie_path=movie_path, threshold=threshold)
    scene_manifest_path = paths.manifest_dir / f"{run_id}.json"
    movie_run_manifest_path = paths.manifest_dir / f"movie_run_{run_id}.json"

    print(f"Movie: {movie_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path.resolve()}")
    print(f"Scene threshold: {threshold:.2f}")

    movie_run_manifest = _load_movie_run_manifest(
        movie_run_manifest_path=movie_run_manifest_path,
        movie_path=movie_path,
        output_path=output_path,
        config_path=config_path,
        threshold=threshold,
        limit=limit,
        resume=resume,
    )
    _mark_movie_stage(movie_run_manifest, stage="scene_detection", status="running")
    write_json_manifest(movie_run_manifest_path, movie_run_manifest)

    try:
        if not overwrite and _can_resume_scene_manifest(
            scene_manifest_path=scene_manifest_path,
            movie_path=movie_path,
            config=config,
            threshold=threshold,
        ):
            print(f"Reusing existing scene manifest: {scene_manifest_path}")
        else:
            run_detect_scenes(
                config=config,
                config_path=config_path,
                movie_path=movie_path,
                output_path=scene_manifest_path,
                threshold=threshold,
            )

        scene_manifest = load_scene_manifest(scene_manifest_path)
        _mark_movie_stage(
            movie_run_manifest,
            stage="scene_detection",
            status="succeeded",
            scene_manifest_path=str(scene_manifest_path),
            scene_count=int(scene_manifest.get("scene_count", len(scene_manifest.get("scenes", [])))),
        )
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)

        _mark_movie_stage(movie_run_manifest, stage="batch_colorize", status="running")
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)
        run_colorize_batch(
            config=config,
            config_path=config_path,
            movie_path=movie_path,
            scene_manifest_path=scene_manifest_path,
            resume=resume,
            limit=limit,
        )
        _mark_movie_stage(movie_run_manifest, stage="batch_colorize", status="succeeded")
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)

        compression_config = config.compression
        assembly_output_path = paths.final_dir / f"{run_id}_assembly_work{output_path.suffix}"

        _mark_movie_stage(movie_run_manifest, stage="assembly", status="running")
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)
        run_assemble_final(
            config=config,
            scene_manifest_path=scene_manifest_path,
            output_path=assembly_output_path,
            limit=limit,
        )
        _mark_movie_stage(movie_run_manifest, stage="assembly", status="succeeded")

        _mark_movie_stage(movie_run_manifest, stage="compression", status="running")
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)
        compression_result = _compress_final_movie(
            input_path=assembly_output_path,
            output_path=output_path,
            source_movie_path=movie_path,
            compression_config=compression_config,
        )
        _mark_movie_stage(
            movie_run_manifest,
            stage="compression",
            status="succeeded",
            **compression_result,
        )

        movie_run_manifest["status"] = "succeeded"
        movie_run_manifest["updated_at"] = utc_now_iso()
        movie_run_manifest["completed_at"] = utc_now_iso()
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)
    except Exception as exc:
        failing_stage = _current_movie_stage(movie_run_manifest)
        if failing_stage is not None:
            _mark_movie_stage(movie_run_manifest, stage=failing_stage, status="failed", error=str(exc))
        movie_run_manifest["status"] = "failed"
        movie_run_manifest["updated_at"] = utc_now_iso()
        movie_run_manifest["error"] = str(exc)
        write_json_manifest(movie_run_manifest_path, movie_run_manifest)
        raise

    if not keep_intermediates:
        _cleanup_movie_artifacts(paths=paths, run_id=run_id)

    print(f"Movie colorization complete: {output_path}")
    return 0


def _resolve_output_path(*, movie_path: Path, output_path: Path | None, limit: int | None) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()

    suffix = "_color"
    if limit is not None:
        suffix = f"{suffix}_first{limit}"
    return movie_path.with_stem(f"{movie_path.stem}{suffix}")


def _build_run_id(*, movie_path: Path, threshold: float) -> str:
    safe_stem = "".join(character if character.isalnum() else "_" for character in movie_path.stem).strip("_")
    location_hash = sha1(str(movie_path).encode("utf-8")).hexdigest()[:8]
    threshold_code = int(round(threshold * 100))
    return f"{safe_stem}_{location_hash}_t{threshold_code:03d}"


def _cleanup_movie_artifacts(*, paths, run_id: str) -> None:
    candidates = [
        paths.scene_dir / run_id,
        paths.colorized_dir / "scenes" / run_id,
        paths.colorized_dir / "deoldify" / run_id,
        paths.colorized_dir / "ddcolor" / run_id,
        paths.final_dir / f"{run_id}_assembly_work.mp4",
        paths.manifest_dir / f"{run_id}.json",
        paths.manifest_dir / f"movie_run_{run_id}.json",
        paths.manifest_dir / f"full_run_{run_id}.json",
        paths.manifest_dir / f"scene_runs_{run_id}.json",
        paths.manifest_dir / f"concat_{run_id}.txt",
        paths.manifest_dir / f"final_assembly_{run_id}.json",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        elif candidate.exists():
            candidate.unlink()


def _load_movie_run_manifest(
    *,
    movie_run_manifest_path: Path,
    movie_path: Path,
    output_path: Path,
    config_path: Path,
    threshold: float,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    if resume:
        payload = load_json_manifest(movie_run_manifest_path, {})
        if isinstance(payload, dict):
            same_movie = payload.get("movie") == str(movie_path)
            same_output = payload.get("output_path") == str(output_path)
            same_threshold = payload.get("scene_threshold") == threshold
            same_limit = payload.get("limit") == limit
            if same_movie and same_output and same_threshold and same_limit:
                payload["updated_at"] = utc_now_iso()
                return payload

    now = utc_now_iso()
    return {
        "run_id": movie_run_manifest_path.stem.removeprefix("movie_run_"),
        "movie": str(movie_path),
        "output_path": str(output_path),
        "config_path": str(config_path.resolve()),
        "scene_threshold": threshold,
        "limit": limit,
        "status": "running",
        "started_at": now,
        "updated_at": now,
        "stages": {
            "scene_detection": {"status": "pending"},
            "batch_colorize": {"status": "pending"},
            "assembly": {"status": "pending"},
            "compression": {"status": "pending"},
        },
    }


def _compress_final_movie(
    *,
    input_path: Path,
    output_path: Path,
    source_movie_path: Path,
    compression_config: dict[str, Any],
) -> dict[str, Any]:
    crfs = [int(compression_config.get("crf", 20))]
    crfs.extend(int(value) for value in compression_config.get("retry_crfs", [23, 26, 28]))
    crfs = list(dict.fromkeys(crfs))

    source_size_bytes = source_movie_path.stat().st_size
    max_size_multiplier = float(compression_config.get("max_size_multiplier", 2.0))
    max_size_bytes = int(source_size_bytes * max_size_multiplier)

    final_size_bytes = 0
    selected_crf = crfs[-1]
    for crf in crfs:
        selected_crf = crf
        compress_video(
            input_path=input_path,
            output_path=output_path,
            video_codec=str(compression_config.get("video_codec", "libx264")),
            preset=str(compression_config.get("preset", "medium")),
            crf=crf,
            audio_codec=str(compression_config.get("audio_codec", "aac")),
            audio_bitrate=str(compression_config.get("audio_bitrate", "160k")),
            faststart=bool(compression_config.get("faststart", True)),
        )
        final_size_bytes = output_path.stat().st_size
        if final_size_bytes <= max_size_bytes:
            break

    if input_path.exists():
        input_path.unlink()

    within_target = final_size_bytes <= max_size_bytes
    if within_target:
        print(f"Compressed final movie written to {output_path} (CRF {selected_crf})")
    else:
        print(
            "Compressed final movie exceeds the configured size target "
            f"after CRF {selected_crf}: {output_path}"
        )

    return {
        "output_path": str(output_path),
        "source_size_bytes": source_size_bytes,
        "final_size_bytes": final_size_bytes,
        "max_size_bytes": max_size_bytes,
        "selected_crf": selected_crf,
        "within_size_target": within_target,
    }


def _mark_movie_stage(
    payload: dict[str, Any],
    *,
    stage: str,
    status: str,
    error: str | None = None,
    **extra: Any,
) -> None:
    stages = payload.setdefault("stages", {})
    stage_payload = stages.setdefault(stage, {})
    stage_payload["status"] = status
    stage_payload["updated_at"] = utc_now_iso()
    if status == "running":
        stage_payload["started_at"] = stage_payload.get("started_at", utc_now_iso())
        stage_payload.pop("error", None)
    if status == "succeeded":
        stage_payload["completed_at"] = utc_now_iso()
        stage_payload.pop("error", None)
    if status == "failed" and error is not None:
        stage_payload["error"] = error
    for key, value in extra.items():
        stage_payload[key] = value
    payload["updated_at"] = utc_now_iso()


def _current_movie_stage(payload: dict[str, Any]) -> str | None:
    for stage_name in ("compression", "assembly", "batch_colorize", "scene_detection"):
        stage_payload = payload.get("stages", {}).get(stage_name, {})
        if stage_payload.get("status") == "running":
            return stage_name
    return None


def _can_resume_scene_manifest(
    *,
    scene_manifest_path: Path,
    movie_path: Path,
    config: AppConfig,
    threshold: float,
) -> bool:
    if not scene_manifest_path.exists():
        return False

    try:
        manifest = load_scene_manifest(scene_manifest_path)
    except Exception:
        return False

    return scene_manifest_matches(
        manifest,
        movie_path=movie_path,
        threshold=threshold,
        min_scene_seconds=float(config.scenes["min_scene_seconds"]),
        max_scene_seconds=float(config.scenes["max_scene_seconds"]),
        overlap_seconds=float(config.scenes["overlap_seconds"]),
    )
