from __future__ import annotations

from dataclasses import dataclass
import os
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
    root = _resolve_runtime_root(config)
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


def _resolve_runtime_root(config: AppConfig) -> Path:
    configured_root = config.raw.get("paths", {}).get("root")
    if configured_root:
        return Path(str(configured_root)).expanduser().resolve()

    env_root = os.environ.get("COLORIT_RUNTIME_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    repo_root = config.path.parent.parent.resolve()
    canonical_root = Path.home() / "ColorIt"
    worktree_root = Path.home() / "worktrees" / "ColorIt"
    try:
        if repo_root.is_relative_to(worktree_root) and canonical_root.exists():
            return canonical_root.resolve()
    except ValueError:
        pass
    return repo_root


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
