from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

from src.pipeline.config import AppConfig
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


MOVIE_PATH = Path("~/Movies/Kannada/emme thammanna.mp4").expanduser()


def run_verify_env(*, config: AppConfig, config_path: Path) -> int:
    paths = resolve_project_paths(config)
    ensure_runtime_directories(paths)

    failures: list[str] = []

    print(f"Config: {config_path.resolve()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"ffmpeg: found at {ffmpeg_path}")
    else:
        failures.append(
            "ffmpeg not found on PATH. Install it first, for example with `brew install ffmpeg`."
        )
        print("ffmpeg: missing")

    if MOVIE_PATH.exists():
        print(f"Source movie: found at {MOVIE_PATH}")
    else:
        print(f"Source movie: not found at {MOVIE_PATH}")

    if paths.weights_path.exists():
        print(f"Model weights: found at {paths.weights_path}")
    else:
        print(f"Model weights: not downloaded yet ({paths.weights_path})")

    print("Model loading: intentionally disabled in this bootstrap phase")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    return 0
