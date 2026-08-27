import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.firered_mlx import FireRedCheckpointError, validate_firered_checkpoint


class FireRedCheckpointValidationTest(unittest.TestCase):
    def _snapshot(self, root: Path, *, zero_cond_t=False) -> None:
        files = {
            "model_index.json": {
                "processor": ["transformers", "Qwen2VLProcessor"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage"],
            },
            "transformer/config.json": {
                "_class_name": "QwenImageTransformer2DModel",
                "attention_head_dim": 128,
                "axes_dims_rope": [16, 56, 56],
                "guidance_embeds": False,
                "in_channels": 64,
                "joint_attention_dim": 3584,
                "num_attention_heads": 24,
                "num_layers": 60,
                "out_channels": 16,
                "patch_size": 2,
                "zero_cond_t": zero_cond_t,
            },
            "transformer/diffusion_pytorch_model.safetensors.index.json": {
                "metadata": {"total_size": 40_860_802_176},
                "weight_map": {f"t{i}": "transformer.safetensors" for i in range(1933)},
            },
            "text_encoder/model.safetensors.index.json": {
                "metadata": {"total_size": 16_584_333_312},
                "weight_map": {f"e{i}": "text_encoder.safetensors" for i in range(729)},
            },
        }
        for relative, value in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        for relative in (
            "transformer/transformer.safetensors",
            "text_encoder/text_encoder.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_accepts_firered_1_1_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._snapshot(root)
            info = validate_firered_checkpoint(root)
            self.assertEqual(info.transformer_keys, 1933)
            self.assertEqual(info.text_encoder_keys, 729)

    def test_rejects_qwen_2511_zero_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._snapshot(root, zero_cond_t=True)
            with self.assertRaisesRegex(FireRedCheckpointError, "zero_cond_t=false"):
                validate_firered_checkpoint(root)


if __name__ == "__main__":
    unittest.main()
