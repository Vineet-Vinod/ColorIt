from pathlib import Path

import pytest
from PIL import Image

from src.pipeline.image_edit_keyframes import (
    IMAGE_EDIT_KEYFRAME_MODELS,
    ImageEditKeyframeColorizer,
    ImageEditKeyframeOptions,
)


def test_options_validate_fixed_mlx_shape() -> None:
    options = ImageEditKeyframeOptions.from_settings({"width": 1024, "height": 576})
    assert options.seed == 101
    assert options.flux_batch_size == 8
    assert options.palette_anchor is True
    with pytest.raises(ValueError, match="multiple of 16"):
        ImageEditKeyframeOptions.from_settings({"width": 1000, "height": 576})
    with pytest.raises(ValueError, match="flux_batch_size"):
        ImageEditKeyframeOptions.from_settings({"flux_batch_size": 0})


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

    single_calls = []
    batch_calls = []

    class FakeRunner:
        def generate(self, **kwargs):
            single_calls.append(kwargs)
            return Image.new("RGB", (64, 64), (20, 40, 60))

        def generate_batch(self, **kwargs):
            batch_calls.append(kwargs)
            return [Image.new("RGB", (64, 64), (20, 40, 60)) for _ in kwargs["source_images"]]

    colorizer = ImageEditKeyframeColorizer(
        model="flux2_klein_4b",
        model_root=tmp_path,
        options=ImageEditKeyframeOptions(width=64, height=64, palette_anchor=True),
    )
    colorizer._runner = FakeRunner()
    colorizer.colorize(sources, outputs)

    assert [call["source_image"] for call in single_calls] == [sources[1]]
    assert single_calls[0]["reference_images"] is None
    assert [call["source_images"] for call in batch_calls] == [[sources[0], sources[2]]]
    assert batch_calls[0]["reference_images"] == [outputs[1]]
    assert batch_calls[0]["prompt"] == colorizer.options.palette_prompt
    assert all(path.is_file() for path in outputs)


def test_flux_keyframes_use_bounded_batches(tmp_path: Path) -> None:
    sources = []
    outputs = []
    for index in range(5):
        source = tmp_path / f"source-{index}.png"
        Image.new("L", (64, 64), 100 + index).save(source)
        sources.append(source)
        outputs.append(tmp_path / f"output-{index}.png")

    calls = []

    class FakeRunner:
        def generate_batch(self, **kwargs):
            calls.append(kwargs)
            return [Image.new("RGB", (64, 64), (20, 40, 60)) for _ in kwargs["source_images"]]

    colorizer = ImageEditKeyframeColorizer(
        model="flux2_klein_4b",
        model_root=tmp_path,
        options=ImageEditKeyframeOptions(
            width=64,
            height=64,
            flux_batch_size=2,
            palette_anchor=False,
        ),
    )
    colorizer._runner = FakeRunner()
    colorizer.colorize(sources, outputs)

    assert [call["source_images"] for call in calls] == [sources[:2], sources[2:4], sources[4:]]
    assert [call["seeds"] for call in calls] == [[101, 101], [101, 101], [101]]
    assert all(path.is_file() for path in outputs)
