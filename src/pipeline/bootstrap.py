from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

from src.pipeline.config import AppConfig
from src.pipeline.ffmpeg_utils import extract_single_frame
from src.pipeline.inference import colorize_image_file
from src.pipeline.model_loader import load_colorizer_bundle
from src.pipeline.paths import ensure_runtime_directories, resolve_project_paths


MOVIE_PATH = Path("~/Movies/Kannada/emme thammanna.mp4").expanduser()


def run_verify_env(
    *,
    config: AppConfig,
    config_path: Path,
    test_image: Path | None = None,
    test_frame_time_seconds: float = 60.0,
    skip_inference: bool = False,
) -> int:
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

    if not paths.weights_path.exists():
        failures.append(f"Model weights not found: {paths.weights_path}")
        print(f"Model weights: missing ({paths.weights_path})")
    else:
        print(f"Model weights: found at {paths.weights_path}")

    try:
        import torch
        import torchvision

        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
    except Exception as exc:
        failures.append(f"Failed to import torch/torchvision: {exc}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    try:
        bundle = load_colorizer_bundle(config)
        print(f"Backend selected: {bundle.backend}")
        print("Model load: succeeded")
    except Exception as exc:
        print(f"ERROR: model load failed: {exc}")
        return 1

    if skip_inference:
        print("Single-frame inference: skipped by request")
        return 0

    frame_input = _resolve_test_image_input(paths=paths, test_image=test_image)
    frame_output = paths.colorized_dir / "verify_env_colorized_frame.png"

    try:
        if test_image is None:
            extract_single_frame(
                input_path=MOVIE_PATH,
                output_path=frame_input,
                time_seconds=test_frame_time_seconds,
            )
            print(f"Verification frame extracted to: {frame_input}")

        colorize_image_file(
            model_bundle=bundle,
            input_path=frame_input,
            output_path=frame_output,
            render_factor=int(config.model["render_factor"]),
            postprocess_config=config.raw["postprocess"],
        )
        print(f"Single-frame inference: succeeded ({frame_output})")
    except Exception as exc:
        print(f"ERROR: single-frame inference failed: {exc}")
        return 1

    return 0


def _resolve_test_image_input(*, paths, test_image: Path | None) -> Path:
    if test_image is not None:
        resolved = test_image.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Test image not found: {resolved}")
        return resolved
    return paths.frames_dir / "verify_env_source_frame.png"
