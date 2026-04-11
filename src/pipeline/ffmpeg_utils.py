from __future__ import annotations

import subprocess
from pathlib import Path


def extract_single_frame(*, input_path: Path, output_path: Path, time_seconds: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(time_seconds),
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    _run(command)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with code {result.returncode}: {result.stderr.strip()}"
        )
