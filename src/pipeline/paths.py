from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline.config import AppConfig


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    scene_dir: Path
    frames_dir: Path
    colorized_dir: Path
    manifest_dir: Path
    final_dir: Path
    weights_path: Path


def resolve_project_paths(config: AppConfig) -> ProjectPaths:
    root = config.path.parent.parent.resolve()
    path_config = config.paths

    return ProjectPaths(
        root=root,
        scene_dir=(root / path_config["scene_dir"]).resolve(),
        frames_dir=(root / path_config["frames_dir"]).resolve(),
        colorized_dir=(root / path_config["colorized_dir"]).resolve(),
        manifest_dir=(root / path_config["manifest_dir"]).resolve(),
        final_dir=(root / path_config["final_dir"]).resolve(),
        weights_path=(root / config.model["weights_path"]).resolve(),
    )


def ensure_runtime_directories(paths: ProjectPaths) -> None:
    for directory in (
        paths.scene_dir,
        paths.frames_dir,
        paths.colorized_dir,
        paths.manifest_dir,
        paths.final_dir,
        paths.weights_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
