from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import torch
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    PerturbationConfig,
)
from ltx_core.model.transformer.attention import AttentionOps, PytorchAttention
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
)
from ltx_core.model.transformer.transformer_args import TransformerArgs

from experiments.adapter_compare.ltx_mlx import (
    MLXBlock,
    from_torch,
    install_mlx_blocks,
    modality_arrays,
)


def arguments(
    dim: int, tokens: int, dtype: torch.dtype, adaln: bool
) -> TransformerArgs:
    heads = 4
    head_dim = dim // heads
    angles = torch.randn(1, heads, tokens, head_dim // 2, dtype=dtype)
    cross_angles = torch.randn(1, heads, tokens, 2, dtype=dtype)
    context = torch.randn(1, 5, dim, dtype=dtype)
    mask = torch.zeros(1, 1, 1, 5, dtype=dtype)
    mask[..., -1] = -10000
    return TransformerArgs(
        x=torch.randn(1, tokens, dim, dtype=dtype),
        context=context,
        context_mask=mask,
        timesteps=torch.randn(1, 1, (9 if adaln else 6) * dim, dtype=dtype) * 0.1,
        embedded_timestep=torch.randn(1, 1, dim, dtype=dtype),
        positional_embeddings=(angles.cos(), angles.sin()),
        cross_positional_embeddings=(cross_angles.cos(), cross_angles.sin()),
        cross_scale_shift_timestep=torch.randn(1, 1, 4 * dim, dtype=dtype) * 0.1,
        cross_gate_timestep=torch.randn(1, 1, dim, dtype=dtype) * 0.1,
        cross_attn_perturbation_mask=torch.ones(1, 1, 1, dtype=dtype),
        prompt_timestep=torch.randn(1, 1, 2 * dim, dtype=dtype) * 0.1
        if adaln
        else None,
        enabled=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("audio_enabled", [False, True])
@pytest.mark.parametrize("adaln", [False, True])
def test_mlx_block_matches_official_joint_attention(
    dtype: torch.dtype,
    audio_enabled: bool,
    adaln: bool,
) -> None:
    torch.manual_seed(17)
    video_config = TransformerConfig(32, 4, 8, 32, True, adaln)
    audio_config = (
        TransformerConfig(16, 4, 4, 16, True, adaln) if audio_enabled else None
    )
    attention = PytorchAttention()
    block = (
        BasicAVTransformerBlock(
            video_config,
            audio_config,
            LTXRopeType.SPLIT,
            attention_ops=AttentionOps(
                attention_function=attention, masked_attention_function=attention
            ),
        )
        .eval()
        .to(dtype)
    )
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.05)
    video = arguments(32, 7, dtype, adaln)
    audio = arguments(16, 3, dtype, adaln) if audio_enabled else None
    with torch.inference_mode():
        expected_video, expected_audio = block(video, audio)
    backend = MLXBlock(block)
    result_video, result_audio = backend(
        from_torch(video.x),
        None if audio is None else from_torch(audio.x),
        modality_arrays(video),
        None if audio is None else modality_arrays(audio),
    )
    assert result_video is not None and expected_video is not None
    mx.eval(result_video, result_audio)
    atol = 0.00001 if dtype == torch.float32 else 0.02
    np.testing.assert_allclose(
        np.array(result_video.astype(mx.float32)),
        expected_video.x.float().numpy(),
        atol=atol,
        rtol=atol,
    )
    if audio_enabled:
        assert result_audio is not None and expected_audio is not None
        np.testing.assert_allclose(
            np.array(result_audio.astype(mx.float32)),
            expected_audio.x.float().numpy(),
            atol=atol,
            rtol=atol,
        )


def test_mlx_reference_attention_honors_additive_mask() -> None:
    torch.manual_seed(5)
    attention = PytorchAttention()
    config = TransformerConfig(32, 4, 8, 32)
    block = BasicAVTransformerBlock(
        config,
        attention_ops=AttentionOps(
            attention_function=attention, masked_attention_function=attention
        ),
    ).eval()
    source = torch.randn(1, 7, 32)
    mask = torch.zeros(1, 1, 7, 7)
    mask[:, :, :3, 3:] = -10000
    backend = MLXBlock(block)
    assert backend.shape.video_attention is not None
    with torch.inference_mode():
        expected = block.attn1(source, mask=mask)
    actual = backend.attention(
        from_torch(source),
        "attn1",
        backend.shape.video_attention,
        mask=from_torch(mask),
    )
    np.testing.assert_allclose(
        np.array(actual), expected.numpy(), atol=0.00001, rtol=0.00001
    )


def test_installed_backend_preserves_joint_block_stack() -> None:
    torch.manual_seed(19)
    attention = PytorchAttention()
    model = LTXModel(
        num_attention_heads=4,
        attention_head_dim=8,
        cross_attention_dim=32,
        audio_num_attention_heads=4,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=16,
        num_layers=2,
        cross_attention_adaln=True,
        attention_ops=AttentionOps(
            attention_function=attention, masked_attention_function=attention
        ),
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.05)
    video, audio = (
        arguments(32, 7, torch.float32, True),
        arguments(16, 3, torch.float32, True),
    )
    perturbations = BatchedPerturbationConfig([PerturbationConfig.empty()], 2)
    with torch.inference_mode():
        expected_video, expected_audio = model._process_transformer_blocks(
            video, audio, perturbations
        )
        backend = install_mlx_blocks(model)
        actual_video, actual_audio = model._process_transformer_blocks(
            video, audio, perturbations
        )
    assert install_mlx_blocks(model) is backend
    assert model.num_blocks == 2
    assert actual_video is not None and actual_audio is not None
    assert expected_video is not None and expected_audio is not None
    torch.testing.assert_close(
        actual_video.x, expected_video.x, atol=0.00001, rtol=0.00001
    )
    torch.testing.assert_close(
        actual_audio.x, expected_audio.x, atol=0.00001, rtol=0.00001
    )
