import unittest

import mlx.core as mx

from src.pipeline.image_edit_weights import IMAGE_EDIT_MODELS
from src.vendor.qwen_mlx.adapter import QWEN_IMAGE_EDIT_2511, QwenEditConfig
from src.vendor.qwen_mlx.qwen_2511 import (
    apply_zero_condition_modulation,
    token_modulation_index,
    zero_condition_timesteps,
)


class QwenEditConfigTest(unittest.TestCase):
    def test_default_is_deterministic_and_uses_quality_quantization(self):
        config = QwenEditConfig()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.quantize, 8)
        self.assertEqual(config.steps, 20)

    def test_rejects_invalid_quantization(self):
        with self.assertRaisesRegex(ValueError, "quantize"):
            QwenEditConfig(quantize=7)

    def test_rejects_non_vae_aligned_dimensions(self):
        with self.assertRaisesRegex(ValueError, "width"):
            QwenEditConfig(width=1001)
        with self.assertRaisesRegex(ValueError, "height"):
            QwenEditConfig(height=0)

    def test_uses_the_shared_pinned_qwen_record(self):
        record = IMAGE_EDIT_MODELS["qwen_image_edit_2511"]
        self.assertEqual(QWEN_IMAGE_EDIT_2511["repo_id"], record.repo_id)
        self.assertEqual(QWEN_IMAGE_EDIT_2511["revision"], record.revision)
        self.assertEqual(QWEN_IMAGE_EDIT_2511["license"], record.license)


class ZeroConditionTimestepTest(unittest.TestCase):
    def test_pairs_denoising_timestep_with_zero_source_timestep(self):
        timestep = mx.array([7.0], dtype=mx.float32)
        actual = zero_condition_timesteps(timestep)
        mx.eval(actual)
        self.assertEqual(actual.tolist(), [7.0, 0.0])

    def test_generated_and_source_tokens_select_different_modulation(self):
        # The tiny fixture uses one channel. Row 0 is the normal denoising
        # timestep for generated tokens. Row 1 is the zero timestep for source
        # image tokens. Each row stores shift, scale, then gate.
        image_tokens = mx.zeros((1, 3, 1), dtype=mx.float32)
        mod_params = mx.array([[2.0, 0.0, 5.0], [11.0, 0.0, 13.0]], dtype=mx.float32)
        index = token_modulation_index(batch_size=1, generated_tokens=2, source_tokens=1)
        modulated, gate = apply_zero_condition_modulation(image_tokens, mod_params, index)
        mx.eval(modulated, gate)

        self.assertEqual(index.tolist(), [[0, 0, 1]])
        self.assertEqual(modulated.tolist(), [[[2.0], [2.0], [11.0]]])
        self.assertEqual(gate.tolist(), [[[5.0], [5.0], [13.0]]])

    def test_rejects_unpaired_timestep_modulation(self):
        with self.assertRaisesRegex(ValueError, "paired timestep"):
            apply_zero_condition_modulation(
                mx.zeros((1, 1, 1)),
                mx.zeros((3, 3)),
                token_modulation_index(1, 1, 0),
            )


if __name__ == "__main__":
    unittest.main()
