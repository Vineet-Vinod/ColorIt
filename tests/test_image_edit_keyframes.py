from pathlib import Path

import pytest
from PIL import Image

from src.pipeline.image_edit_keyframes import (
    IMAGE_EDIT_KEYFRAME_MODELS,
    ImageEditKeyframeColorizer,
    ImageEditKeyframeOptions,
)


def test_options_validate_fixed_mlx_shape() -> None:
    assert ImageEditKeyframeOptions.from_settings({"width": 1024, "height": 576}).seed == 101
    with pytest.raises(ValueError, match="multiple of 16"):
        ImageEditKeyframeOptions.from_settings({"width": 1000, "height": 576})


def test_only_promoted_editor_is_a_public_keyframe_model(tmp_path: Path) -> None:
    assert IMAGE_EDIT_KEYFRAME_MODELS == ("flux2_klein_4b",)
    with pytest.raises(ValueError, match="Unsupported MLX image editor"):
        ImageEditKeyframeColorizer(
            model="qwen_image_edit_2511",
            model_root=tmp_path,
            options=ImageEditKeyframeOptions(),
        )


def test_flux_palette_bank_colors_anchor_first(tmp_path: Path) -> None:
    sources = []
    outputs = []
    for index in range(3):
        source = tmp_path / f"source-{index}.png"
        Image.new("L", (64, 64), 100 + index).save(source)
        sources.append(source)
        outputs.append(tmp_path / f"output-{index}.png")

    calls = []

    class FakeRunner:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return Image.new("RGB", (64, 64), (20, 40, 60))

    colorizer = ImageEditKeyframeColorizer(
        model="flux2_klein_4b",
        model_root=tmp_path,
        options=ImageEditKeyframeOptions(width=64, height=64, palette_anchor=True),
    )
    colorizer._runner = FakeRunner()
    colorizer.colorize(sources, outputs)

    assert [call["source_image"] for call in calls] == [sources[1], sources[0], sources[2]]
    assert calls[0]["reference_images"] is None
    assert calls[1]["reference_images"] == [outputs[1]]
    assert all(path.is_file() for path in outputs)
