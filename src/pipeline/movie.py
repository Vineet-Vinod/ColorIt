from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import shutil

from src.pipeline.assemble import run_assemble_final
from src.pipeline.batch import run_colorize_batch
from src.pipeline.config import AppConfig
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths
from src.pipeline.scenes import run_detect_scenes


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

    print(f"Movie: {movie_path}")
    print(f"Output: {output_path}")
    print(f"Config: {config_path.resolve()}")
    print(f"Scene threshold: {threshold:.2f}")

    run_detect_scenes(
        config=config,
        config_path=config_path,
        movie_path=movie_path,
        output_path=scene_manifest_path,
        threshold=threshold,
    )
    run_colorize_batch(
        config=config,
        config_path=config_path,
        movie_path=movie_path,
        scene_manifest_path=scene_manifest_path,
        resume=resume,
        limit=limit,
    )
    run_assemble_final(
        config=config,
        scene_manifest_path=scene_manifest_path,
        output_path=output_path,
        limit=limit,
    )

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
        paths.manifest_dir / f"{run_id}.json",
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
