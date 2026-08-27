from __future__ import annotations

import pytest

from src.cli import build_parser


def test_download_weights_accepts_repeatable_image_edit_models() -> None:
    args = build_parser().parse_args(
        [
            "download-weights",
            "--skip-core",
            "--image-edit-model",
            "flux2_klein_4b",
            "--image-edit-model",
            "firered_image_edit_1_1",
        ]
    )

    assert args.skip_core is True
    assert args.image_edit_model == ["flux2_klein_4b", "firered_image_edit_1_1"]


def test_download_weights_rejects_unknown_image_edit_model() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["download-weights", "--image-edit-model", "untrusted/model"]
        )
