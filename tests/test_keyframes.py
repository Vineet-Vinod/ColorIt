from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline.keyframes import _write_source_luma_reference


def test_source_luma_reference_keeps_generated_chroma_and_raw_image(tmp_path: Path) -> None:
    source = np.tile(np.linspace(64, 192, 64, dtype=np.uint8), (64, 1))
    source_bgr = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    colored_bgr = np.full((32, 32, 3), (160, 40, 180), dtype=np.uint8)
    source_path = tmp_path / "source.png"
    colored_path = tmp_path / "colored.png"
    reference_path = tmp_path / "references" / "reference.png"
    assert cv2.imwrite(str(source_path), source_bgr)
    assert cv2.imwrite(str(colored_path), colored_bgr)
    colored_before = colored_path.read_bytes()

    _write_source_luma_reference(
        source_path=source_path,
        colored_path=colored_path,
        output_path=reference_path,
    )

    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    assert reference is not None
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    colored_resized = cv2.resize(colored_bgr, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    colored_lab = cv2.cvtColor(colored_resized, cv2.COLOR_BGR2LAB)
    reference_luma_error = np.abs(
        reference_lab[:, :, 0].astype(float) - source_lab[:, :, 0].astype(float)
    ).mean()
    colored_luma_error = np.abs(
        colored_lab[:, :, 0].astype(float) - source_lab[:, :, 0].astype(float)
    ).mean()
    assert reference_luma_error < colored_luma_error * 0.25
    assert np.abs(
        reference_lab[:, :, 1:].astype(float) - colored_lab[:, :, 1:].astype(float)
    ).mean() < 8.0
    assert reference_lab[:, :, 1:].std(axis=(0, 1)).max() <= 3.0
    assert colored_path.read_bytes() == colored_before


def test_source_luma_reference_rejects_large_geometry_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.png"
    colored_path = tmp_path / "colored.png"
    output_path = tmp_path / "reference.png"
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(source_path), image)
    assert cv2.imwrite(str(colored_path), image)

    class FakeFlow:
        def setUseSpatialPropagation(self, _enabled: bool) -> None:
            return None

        def calc(self, source, _colored, _initial):
            return np.full((*source.shape, 2), 20.0, dtype=np.float32)

    monkeypatch.setattr(cv2, "DISOpticalFlow_create", lambda _preset: FakeFlow())
    with pytest.raises(RuntimeError, match="changed the source geometry"):
        _write_source_luma_reference(
            source_path=source_path,
            colored_path=colored_path,
            output_path=output_path,
        )
    assert not output_path.exists()
