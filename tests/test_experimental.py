from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.cli import main
from src.experimental.assets import verify_artifact
from src.experimental.clip import run_experimental_clip
from src.experimental.experimental_dataclasses import Artifact, ClipInfo, ClipRequest
from src.experimental.media import (
    deliver_clip,
    detect_shots,
    make_shots,
    probe_clip,
    restore_luminance,
)


@pytest.mark.parametrize("duration", [60.0, 60.01, 120.0])
def test_rejects_long_clips_before_model_setup(
    duration: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "src.experimental.clip.probe_clip",
        lambda _: ClipInfo(64, 64, Fraction(25), 1500, duration, False),
    )
    prepare = Mock()
    monkeypatch.setattr("src.experimental.clip.prepare_assets", prepare)
    with pytest.raises(ValueError, match="shorter than 60"):
        run_experimental_clip(ClipRequest(source=tmp_path / "source.mp4"), tmp_path)
    prepare.assert_not_called()


@pytest.mark.parametrize("case", ["input", "existing", "resume"])
def test_rejects_unsafe_output_and_unsupported_resume(
    case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = source if case == "input" else tmp_path / "result.mp4"
    output.write_bytes(b"existing")
    monkeypatch.setattr(
        "src.experimental.clip.probe_clip",
        lambda _: ClipInfo(64, 64, Fraction(25), 25, 1, False),
    )
    request = ClipRequest(source=source, output=output, resume=case == "resume")
    with pytest.raises((ValueError, FileExistsError)):
        run_experimental_clip(request, tmp_path)
    assert output.read_bytes() == b"existing"


def test_regular_cli_keeps_legacy_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = Mock(return_value=0)
    experimental = Mock(return_value=0)
    monkeypatch.setattr("src.cli.run_movie", legacy)
    monkeypatch.setattr("src.experimental.clip.run_experimental_clip", experimental)
    assert main(["colorize-movie", "--input", "movie.mp4"]) == 0
    legacy.assert_called_once()
    experimental.assert_not_called()


def test_experimental_cli_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    experimental = Mock(return_value=0)
    monkeypatch.setattr("src.experimental.clip.run_experimental_clip", experimental)
    assert main(["colorize-movie", "--input", "clip.mp4", "--experimental"]) == 0
    request, _ = experimental.call_args.args
    assert request.source == Path("clip.mp4")


def test_reference_frames_stay_inside_their_shots() -> None:
    shots = make_shots([0, 1, 4, 4, 10, -1, 100], 10)
    assert [(shot.start, shot.end) for shot in shots] == [(0, 1), (1, 4), (4, 10)]
    assert shots[0].references == (0,)
    for shot in shots:
        assert shot.references
        assert all(shot.start <= frame < shot.end for frame in shot.references)


def test_detects_the_missed_dance_cut_without_a_one_frame_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = "pts_time:0.04\nlavfi.scene_score=0.108953\npts_time:5.64\nlavfi.scene_score=0.223589\npts_time:5.68\nlavfi.scene_score=0.136544\n"
    monkeypatch.setattr(
        "src.experimental.media.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout=metadata)),
    )
    info = ClipInfo(1920, 1080, Fraction(25), 300, 12, True)
    shots = detect_shots(Path("clip.mp4"), info)
    assert [(shot.start, shot.end) for shot in shots] == [(0, 141), (141, 300)]


def test_predicted_gray_preserves_source_brightness() -> None:
    source: NDArray[np.uint8] = np.full((24, 32, 3), 160, dtype=np.uint8)
    result = restore_luminance(source, Image.new("RGB", (8, 8), (20, 20, 20)))
    np.testing.assert_allclose(result, source, atol=1)


def test_model_checksum_rejects_same_size_corruption(tmp_path: Path) -> None:
    artifact = Artifact(
        path="model",
        url="https://example.test/model",
        size=4,
        sha256=hashlib.sha256(b"good").hexdigest(),
    )
    file = tmp_path / "model"
    file.write_bytes(b"evil")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_artifact(file, artifact)


@pytest.mark.parametrize("audio", [True, False])
def test_delivery_preserves_fractional_fps_frames_and_audio(
    tmp_path: Path, audio: bool
) -> None:
    source = tmp_path / "source.mkv"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=30000/1001",
    ]
    if audio:
        command.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
    command.extend(["-t", "1.001", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)])
    subprocess.run(command, check=True)
    info = probe_clip(source, count_frames=True)
    output = tmp_path / "result.mp4"
    deliver_clip(source, source, output, info)
    result = probe_clip(output, count_frames=True)
    assert result.frames == info.frames == 30
    assert result.fps == Fraction(30000, 1001)
    assert result.audio == audio
    assert output.stat().st_size <= source.stat().st_size * 2
