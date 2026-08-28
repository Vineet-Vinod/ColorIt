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


if __name__ == "__main__":
    unittest.main()
