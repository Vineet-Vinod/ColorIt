from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.pipeline.semantic_material_lock import MaterialLockSettings, lock_material_chroma


def test_material_lock_removes_spill_without_recoloring_protected_person() -> None:
    source = np.full((32, 32, 3), 170, dtype=np.uint8)
    color = source.copy()
    color[:, :] = (80, 40, 190)
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[4:28, 4:28] = 18
    labels[11:21, 11:21] = 12

    output, alpha = lock_material_chroma(
        source_bgr=source,
        color_bgr=color,
        semantic_labels=[labels],
        material_label=18,
        material_color="#D8CBAE",
        protected_labels=[12],
        settings=MaterialLockSettings(
            close_radius=0,
            expand_radius=0,
            protect_radius=0,
            feather_radius=0,
        ),
    )

    output_lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.int16)
    target = cv2.cvtColor(np.array([[[174, 203, 216]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    protected_lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    protected_lab[:, :, 0] = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0]
    protected_expected = cv2.cvtColor(protected_lab, cv2.COLOR_LAB2BGR)
    protected_expected_lab = cv2.cvtColor(protected_expected, cv2.COLOR_BGR2LAB).astype(np.int16)
    assert np.max(np.abs(output_lab[6, 6, 1:3] - target[1:3])) <= 1
    assert np.max(np.abs(output_lab[16, 16, 1:3] - protected_expected_lab[16, 16, 1:3])) <= 1
    assert alpha[6, 6] == 1.0
    assert alpha[16, 16] == 0.0


def test_material_lock_preserves_source_luma() -> None:
    source = np.tile(np.arange(24, dtype=np.uint8)[:, None, None], (1, 24, 3)) * 9
    color = np.full((24, 24, 3), (20, 40, 210), dtype=np.uint8)
    labels = np.full((24, 24), 18, dtype=np.uint8)
    output, _ = lock_material_chroma(
        source_bgr=source,
        color_bgr=color,
        semantic_labels=[labels],
        material_label=18,
        material_color="#D8CBAE",
    )
    source_luma = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.int16)
    output_luma = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.int16)
    assert np.max(np.abs(source_luma - output_luma)) <= 2


def test_material_lock_rejects_invalid_inputs() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    labels = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="RRGGBB"):
        lock_material_chroma(
            source_bgr=frame,
            color_bgr=frame,
            semantic_labels=[labels],
            material_label=18,
            material_color="cream",
        )
    with pytest.raises(ValueError, match="At least one"):
        lock_material_chroma(
            source_bgr=frame,
            color_bgr=frame,
            semantic_labels=[],
            material_label=18,
            material_color="#D8CBAE",
        )
