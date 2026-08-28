from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from src.pipeline.bonsai_image_mlx import BonsaiImageMLXColorizer, BonsaiImageOptions


class _Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded = 0
        self.calls = []

    def load_transformer_and_vae(self):
        self.loaded += 1

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(image=Image.new("RGB", (64, 64), "red"))


class BonsaiImageMLXTest(unittest.TestCase):
    def test_requires_distilled_settings(self):
        with self.assertRaisesRegex(ValueError, "four"):
            BonsaiImageOptions(variant="binary", steps=5)
        with self.assertRaisesRegex(ValueError, "guidance"):
            BonsaiImageOptions(variant="ternary", guidance=2.0)
        with self.assertRaisesRegex(ValueError, "image_strength"):
            BonsaiImageOptions(variant="ternary", image_strength=0.0)

    def test_reference_conditioned_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (64, 64)).save(source)
            made = []

            def factory(**kwargs):
                model = _Model(**kwargs)
                made.append(model)
                return model

            colorizer = BonsaiImageMLXColorizer(
                root,
                BonsaiImageOptions(variant="ternary"),
                model_factory=factory,
            )
            with patch("src.pipeline.bonsai_image_mlx.verify_model_snapshot"):
                result = colorizer.generate(
                    source_image=source,
                    prompt="Colorize only",
                    seed=101,
                    width=64,
                    height=64,
                )
            self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(made[0].kwargs["precision"], "2bit")
            self.assertEqual(made[0].loaded, 1)
            self.assertEqual(made[0].calls[0]["image_paths"], [source.resolve()])
            self.assertEqual(made[0].calls[0]["num_inference_steps"], 4)
            self.assertEqual(colorizer.kernel_mode, "prism_native")

    def test_img2img_conditioning_uses_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (64, 64)).save(source)
            model = _Model()
            colorizer = BonsaiImageMLXColorizer(
                root,
                BonsaiImageOptions(variant="ternary", conditioning="img2img", image_strength=0.625),
                model_factory=lambda **_kwargs: model,
            )
            with patch("src.pipeline.bonsai_image_mlx.verify_model_snapshot"):
                colorizer.generate(
                    source_image=source,
                    prompt="Colorize only",
                    seed=101,
                    width=64,
                    height=64,
                )
            self.assertEqual(model.calls[0]["image_path"], source.resolve())
            self.assertEqual(model.calls[0]["image_strength"], 0.625)
            self.assertNotIn("image_paths", model.calls[0])


if __name__ == "__main__":
    unittest.main()
