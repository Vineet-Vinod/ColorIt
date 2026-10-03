from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from experiments.adapter_compare.havc_mlx import (
    from_torch,
    install_mlx_transformer,
    read_lora,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "tmp/adapter_compare_90_120/upstream/HAVCServerDiT"


def test_nonfinite_havc_stops_before_decode() -> None:
    from experiments.adapter_compare.havc_numerics import guard_method

    class Decoder:
        called = False

        def decode(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
            self.called = True
            return samples["samples"]

    decoder = Decoder()
    guard_method(decoder, "decode", "vae.decode")
    with pytest.raises(FloatingPointError, match="vae.decode.input"):
        decoder.decode({"samples": torch.tensor([float("nan")])})
    assert not decoder.called


def test_nonfinite_havc_encoding_is_rejected() -> None:
    from experiments.adapter_compare.havc_numerics import guard_method

    encoder = SimpleNamespace(encode=lambda: [[torch.tensor([float("inf")]), {}]])
    guard_method(encoder, "encode", "clip")
    with pytest.raises(FloatingPointError, match="clip.output"):
        encoder.encode()


@pytest.fixture(scope="module", autouse=True)
def cpu_only() -> None:
    mx.set_default_device(mx.cpu)
    torch.set_num_threads(2)
    pytest.importorskip(
        "comfy_kitchen",
        reason="HAVC native runtime is optional; use havc_env for parity",
    )
    pytest.importorskip(
        "comfy_aimdo", reason="HAVC native runtime is optional; use havc_env for parity"
    )
    if not UPSTREAM.is_dir():
        pytest.skip("HAVC upstream source is required for independent native parity")
    sys.path.insert(0, str(UPSTREAM))
    importlib.import_module("comfy_bridge")
    from comfy.cli_args import args

    args.cpu = True
    args.disable_dynamic_vram = True
    args.use_pytorch_cross_attention = True


@pytest.fixture
def native_model():
    import comfy.model_management
    import comfy.ops
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

    # Native unfused functions provide an independent PyTorch CPU reference.
    comfy.model_management.in_training = True
    torch.manual_seed(97)
    model = QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        context_in_dim=10,
        axes_dims_rope=(2, 2, 4),
        operations=comfy.ops.disable_weight_init,
        dtype=torch.float32,
        device="cpu",
    )
    for name, parameter in model.named_parameters():
        with torch.no_grad():
            if ".norm_q." in name or ".norm_k." in name:
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.1)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.parametrize("refs_count", [0, 2])
def test_full_native_transformer_parity(
    native_model, tmp_path: Path, dtype: torch.dtype, with_lora: bool, refs_count: int
) -> None:
    from viggle_turbo import run_with_lora

    model = native_model.to(dtype)
    x = torch.randn(2, 4, 2, 3, dtype=dtype)
    context = torch.randn(2, 5, 10, dtype=dtype)
    refs = [torch.randn(2, 4, 3, 2, dtype=dtype), torch.randn(2, 4, 1, 3, dtype=dtype)][
        :refs_count
    ]
    slots = [1, 3][:refs_count]
    timesteps = torch.tensor([0.8123, 0.1879], dtype=dtype)
    lora = {}
    lora_path = None
    if with_lora:
        state = {}
        for index, block in enumerate(model.transformer_blocks):
            targets = [
                ("attn.to_" + part, block.attn.get_submodule("to_" + part))
                for part in ("q", "k", "v", "out.0")
            ]
            # Viggle's fused gates have virtual, unfused LoRA target names.
            hidden = block.img_mlp.out.in_features
            targets += [
                (
                    "img_mlp.gate_layer",
                    SimpleNamespace(in_features=16, out_features=hidden),
                ),
                ("img_mlp.proj", SimpleNamespace(in_features=16, out_features=hidden)),
                ("img_mlp.out", block.img_mlp.out),
            ]
            for name, module in targets:
                name = f"transformer_blocks.{index}.{name}"
                a = torch.randn(3, module.in_features, dtype=dtype) * 0.1
                b = torch.randn(module.out_features, 3, dtype=dtype) * 0.1
                lora[name] = [a, b * 0.75]
                state[f"transformer.{name}.lora_A.weight"] = a
                state[f"transformer.{name}.lora_B.weight"] = b
        lora_path = tmp_path / "adapter.safetensors"
        save_file(
            state,
            str(lora_path),
            metadata={
                "lora_adapter_metadata": json.dumps(
                    {"transformer.lora_alpha": 3, "transformer.r": 4}
                )
            },
        )

    class Executor:
        class_obj = model

        def __call__(self, *args, **kwargs):
            return model._forward(*args, **kwargs)

    with torch.inference_mode():
        inputs = [
            (x, timesteps),
            (x + 0.2, timesteps * 0.8),
            (x - 0.1, timesteps * 0.5),
        ]
        expected = [
            run_with_lora(lora, Executor(), target, step, context, refs, slots)
            if with_lora
            else model._forward(target, step, context, refs, slots)
            for target, step in inputs
        ]
        backend = install_mlx_transformer(model, lora_path)
        actual = [
            model._forward(target, step, context, refs, slots)
            for target, step in inputs
        ]
    atol = 2e-6 if dtype == torch.float32 else 0.01
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            output.float().numpy(), reference.float().numpy(), atol=atol, rtol=atol
        )
    assert all(parameter.numel() == 0 for parameter in model.parameters())
    assert backend.shape.layers == 2
    assert backend.prefix_cache_hits == 2
    assert backend.prefix_cache_misses == 1


def test_prefix_does_not_see_later_target(native_model) -> None:
    model = native_model
    backend = install_mlx_transformer(model)
    x = from_torch(torch.randn(1, 4, 2, 3))
    context = from_torch(torch.randn(1, 5, 10))
    refs = [from_torch(torch.randn(1, 4, 3, 2))]
    hidden, rope, segments = backend.build_sequence(x, context, refs, [2])
    prefix = hidden.shape[1] - 6
    _, *mod = backend.modulation(mx.array([0.8]), mx.float32)
    changed = mx.concatenate([hidden[:, :prefix], hidden[:, prefix:] + 100], axis=1)
    original = backend.block(hidden, rope, segments, tuple(mod), prefix, 0)
    altered = backend.block(changed, rope, segments, tuple(mod), prefix, 0)
    mx.eval(original, altered)
    np.testing.assert_array_equal(
        np.array(original[:, :prefix]), np.array(altered[:, :prefix])
    )
    assert not np.allclose(
        np.array(original[:, prefix:]), np.array(altered[:, prefix:])
    )


def test_lora_metadata_scales_separate_b_in_bfloat16(tmp_path: Path) -> None:
    a, b = (
        torch.ones(2, 3, dtype=torch.bfloat16),
        torch.full((4, 2), 0.123, dtype=torch.bfloat16),
    )
    path = tmp_path / "adapter.safetensors"
    save_file(
        {"transformer.img_in.lora_A.weight": a, "transformer.img_in.lora_B.weight": b},
        str(path),
        metadata={
            "lora_adapter_metadata": json.dumps(
                {"transformer.lora_alpha": 5, "transformer.r": 3}
            )
        },
    )
    actual = read_lora(path, 0.7)["img_in"]
    np.testing.assert_array_equal(
        np.array(actual.up.astype(mx.float32)), (b * (0.7 * 5 / 3)).float().numpy()
    )


@pytest.mark.parametrize(
    "change",
    ["reference", "context", "slots", "shape", "context_dtype", "reference_dtype"],
)
def test_prefix_cache_invalidates_exact_inputs(native_model, change: str) -> None:
    model = native_model
    x = torch.randn(1, 4, 2, 3)
    context = torch.randn(1, 5, 10)
    reference = torch.randn(1, 4, 3, 2)
    if change.endswith("_dtype"):
        context = context.round()
        reference = reference.round()
    backend = install_mlx_transformer(model)
    with torch.inference_mode():
        model._forward(x, torch.tensor([0.8]), context, [reference], [2])
        # Mutate the same Torch storage to catch aliases in cached key tensors.
        if change == "reference":
            reference.add_(0.123)
        elif change == "context":
            context.add_(0.123)
        elif change == "shape":
            x = x.transpose(-1, -2)
        elif change == "context_dtype":
            context = context.to(torch.bfloat16)
        elif change == "reference_dtype":
            reference = reference.to(torch.bfloat16)
        slots = [3] if change == "slots" else [2]
        actual = model._forward(x, torch.tensor([0.5]), context, [reference], slots)
        assert backend.prefix_cache_hits == 0
        assert backend.prefix_cache_misses == 2
        backend.prefix_cache_enabled = False
        expected = model._forward(x, torch.tensor([0.5]), context, [reference], slots)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-6, rtol=2e-6)


def test_two_cfg_slots_reuse_and_native_reset(native_model) -> None:
    model = native_model
    backend = install_mlx_transformer(model)
    x = torch.randn(1, 4, 2, 3)
    positive = torch.randn(1, 5, 10)
    negative = torch.randn(1, 3, 10)
    with torch.inference_mode():
        for step in (0.8, 0.6, 0.4):
            for context in (positive, negative):
                model._forward(x, torch.tensor([step]), context)
    assert backend.prefix_cache_hits == 4
    assert backend.prefix_cache_misses == 2
    assert len(backend.prefix_cache) == 2
    model.reset_prefix_cache(False)
    assert not backend.prefix_cache_enabled
    assert not backend.prefix_cache
