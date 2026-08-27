import unittest
from importlib.util import find_spec

import mlx.core as mx

from src.pipeline.image_edit_weights import IMAGE_EDIT_MODELS
from src.vendor.qwen_mlx.adapter import QWEN_IMAGE_EDIT_2511, QwenEditConfig
from src.vendor.qwen_mlx.qwen_2511 import (
    apply_zero_condition_modulation,
    token_modulation_index,
    zero_condition_timesteps,
    enable_zero_cond_t,
    zero_cond_transformer_forward,
)
from src.vendor.qwen_mlx.optimized import (
    CompiledQwenCFG,
    PreparedQwenEdit,
    QwenCompiledEditLoop,
    benchmark_fixed_shape,
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


class CompiledQwenCFGTest(unittest.TestCase):
    @staticmethod
    def _toy_transformer(_transformer, timestep, _config, hidden_states, prompt, _mask, _grid):
        # A tiny stand-in for the expensive Qwen transformer. It keeps the
        # timestep dynamic and makes positive versus negative CFG observable.
        return hidden_states + prompt[:, :1, :] + timestep

    def _cfg(self, *, batch_size, mode="auto"):
        static = mx.zeros((batch_size, 2, 1), dtype=mx.float32)
        positive = mx.full((batch_size, 1, 1), 2.0, dtype=mx.float32)
        negative = mx.full((batch_size, 2, 1), -1.0, dtype=mx.float32)
        positive_mask = mx.ones((batch_size, 1), dtype=mx.float32)
        negative_mask = mx.ones((batch_size, 2), dtype=mx.float32)
        return CompiledQwenCFG(
            transformer=object(),
            config=object(),
            static_image_latents=static,
            positive_prompt_embeds=positive,
            positive_prompt_mask=positive_mask,
            negative_prompt_embeds=negative,
            negative_prompt_mask=negative_mask,
            cond_image_grid=(1, 1, 2),
            guidance=1.0,
            cfg_mode=mode,
            fused_cfg_max_batch=1,
            transformer_call=self._toy_transformer,
        )

    def test_fused_cfg_compiles_once_and_keeps_timestep_dynamic(self):
        cfg = self._cfg(batch_size=1)
        latents = mx.zeros((1, 3, 1), dtype=mx.float32)

        first = cfg.predict_noise(latents, mx.array(1.0))
        second = cfg.predict_noise(latents, mx.array(4.0))
        mx.eval(first, second)

        self.assertEqual(cfg.stats.fused_graph_builds, 1)
        self.assertEqual(cfg.stats.denoise_steps, 2)
        self.assertEqual(first.tolist(), [[[3.0], [3.0], [3.0]]])
        self.assertEqual(second.tolist(), [[[6.0], [6.0], [6.0]]])

    def test_auto_uses_two_compiled_graphs_for_memory_safe_multi_seed_batch(self):
        cfg = self._cfg(batch_size=2)
        latents = mx.zeros((2, 3, 1), dtype=mx.float32)
        output = cfg.predict_noise(latents, mx.array(1.0))
        mx.eval(output)

        self.assertEqual(cfg.stats.positive_graph_builds, 1)
        self.assertEqual(cfg.stats.negative_graph_builds, 1)
        self.assertEqual(cfg.stats.fused_graph_builds, 0)
        self.assertEqual(output.shape, (2, 3, 1))

    def test_synthetic_benchmark_reuses_one_fixed_shape_graph(self):
        cfg = self._cfg(batch_size=1)
        result = benchmark_fixed_shape(
            cfg,
            mx.zeros((1, 3, 1), dtype=mx.float32),
            [mx.array(1.0), mx.array(2.0), mx.array(3.0)],
        )
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["fused_graph_builds"], 1)
        self.assertEqual(result["positive_graph_builds"], 0)
        self.assertGreaterEqual(result["seconds"], 0.0)


class _SyntheticScheduler:
    def __init__(self):
        self.timesteps = [mx.array(1.0), mx.array(2.0)]
        self.sigmas = [mx.array(0.25), mx.array(0.5)]

    @staticmethod
    def step(*, noise, timestep, latents):
        return latents + (0.1 + timestep * 0.01) * noise


class CompiledDenoiseParityTest(unittest.TestCase):
    def test_compiled_loop_matches_synthetic_eager_scheduler(self):
        cfg = CompiledQwenCFGTest()._cfg(batch_size=1)
        config = type("Config", (), {"scheduler": _SyntheticScheduler(), "num_inference_steps": 2})()
        prepared = PreparedQwenEdit(cfg=cfg, config=config, batch_size=1)
        initial = mx.zeros((1, 3, 1), dtype=mx.float32)
        actual = QwenCompiledEditLoop(model=object()).denoise_latents(prepared, initial)

        eager = initial
        for index, timestep in enumerate(config.scheduler.sigmas):
            positive = eager + 2.0 + timestep
            negative = eager - 1.0 + timestep
            noise = CompiledQwenCFG._guided_noise(positive, negative, 1.0)
            eager = config.scheduler.step(noise=noise, timestep=index, latents=eager)
        mx.eval(actual, eager)
        self.assertEqual(actual.tolist(), eager.tolist())

    def test_batch_unpack_keeps_each_seed_separate(self):
        # This is the same reshape/transpose order as mflux's B=1 unpacker,
        # extended over a real batch dimension.
        packed = mx.arange(2 * 4 * 64, dtype=mx.float32).reshape((2, 4, 64))
        unpacked = QwenCompiledEditLoop._unpack_batch(packed, height=32, width=32)
        mx.eval(unpacked)
        self.assertEqual(unpacked.shape, (2, 16, 4, 4))
        self.assertNotEqual(unpacked[0].tolist(), unpacked[1].tolist())


@unittest.skipUnless(find_spec("mflux"), "mflux is optional for unit tests")
class MfluxMinimalParityTest(unittest.TestCase):
    def test_compiled_2511_transformer_matches_eager_minimal_fixture(self):
        from types import SimpleNamespace
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer

        mx.random.seed(7)
        transformer = QwenTransformer(
            num_layers=1,
            num_attention_heads=1,
            attention_head_dim=128,
            joint_attention_dim=8,
        )
        enable_zero_cond_t(SimpleNamespace(transformer=transformer))
        config = SimpleNamespace(
            width=32,
            height=32,
            scheduler=SimpleNamespace(sigmas=mx.array([1.0]), timesteps=mx.array([1000])),
        )
        hidden = mx.zeros((1, 5, 64), dtype=mx.float32)
        prompt = mx.zeros((1, 3, 8), dtype=mx.float32)
        mask = mx.ones((1, 3), dtype=mx.float32)
        eager = transformer(0, config, hidden, prompt, mask, cond_image_grid=(1, 1, 1))
        compiled = mx.compile(
            lambda timestep: zero_cond_transformer_forward(
                transformer, timestep, config, hidden, prompt, mask, (1, 1, 1)
            )
        )(mx.array(1.0))
        mx.eval(eager, compiled)
        max_difference = mx.max(mx.abs(eager - compiled)).item()
        self.assertLess(max_difference, 5e-5)


if __name__ == "__main__":
    unittest.main()
