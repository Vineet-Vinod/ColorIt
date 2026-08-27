from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.pipeline import flux2_klein_mlx as flux2


def test_manifest_uses_immutable_official_revision() -> None:
    assert flux2.FLUX2_KLEIN_4B_REVISION == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert flux2.MFLUX_GIT_REVISION == "e3d313ec2191f1960b3c9f3745299dcc58d2fdeb"
    assert flux2.flux2_klein_4b_required_bytes() == 15_968_709_091
    assert all(item.url.startswith("https://huggingface.co/black-forest-labs/") for item in flux2.FLUX2_KLEIN_4B_FILES)
    assert all(len(item.sha256) == 64 for item in flux2.FLUX2_KLEIN_4B_FILES)


def test_options_reject_unsupported_distilled_settings() -> None:
    with pytest.raises(ValueError, match="guidance"):
        flux2.Flux2KleinMLXOptions(guidance=1.1)
    with pytest.raises(ValueError, match="KV-cache"):
        flux2.Flux2KleinMLXOptions(use_kv_cache=True)


def test_generate_reuses_model_and_passes_fast_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.png"
    Image.new("L", (64, 64), color=128).save(source)
    calls: list[dict[str, object]] = []

    class FakeModel:
        def generate_image(self, **kwargs):
            calls.append(kwargs)
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(1, 2, 3))

    monkeypatch.setattr(flux2, "verify_flux2_klein_4b_weights", lambda _path: None)
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return FakeModel()

    colorizer = flux2.Flux2KleinMLXColorizer(tmp_path, model_factory=factory)
    colorizer.warmup()
    result = colorizer.generate(
        source_image=source,
        reference_images=[source],
        prompt="Color this black and white film still naturally.",
        seed=7,
        width=64,
        height=64,
    )

    assert result.mode == "RGB"
    assert factory_calls == [{"model_path": str(tmp_path.resolve()), "quantize": 8}]
    assert calls == [
        {
            "seed": 7,
            "prompt": "Color this black and white film still naturally.",
            "num_inference_steps": 4,
            "width": 64,
            "height": 64,
            "guidance": 1.0,
            "image_paths": [source.resolve(), source.resolve()],
            "scheduler": "flow_match_euler_discrete",
            "use_kv_cache": False,
        }
    ]
