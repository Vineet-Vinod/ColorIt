from __future__ import annotations

from dataclasses import replace
from types import MethodType
from typing import cast

import mlx.core as mx
import numpy as np
import torch
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.model import LTXModel
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import TransformerArgs
from mlx import nn

from .adapter_compare_dataclasses import AttentionShape, BlockShape, ModalityArrays


def from_torch(value: torch.Tensor) -> mx.array:
    return mx.from_dlpack(value.detach().to("cpu").contiguous())


def to_torch(value: mx.array, reference: torch.Tensor) -> torch.Tensor:
    mx.eval(value)
    array = np.array(value.astype(mx.float32))
    return torch.from_numpy(array).to(device=reference.device, dtype=reference.dtype)


def optional_array(value: torch.Tensor | None) -> mx.array | None:
    return None if value is None else from_torch(value)


def optional_rope(
    value: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[mx.array, mx.array] | None:
    if value is None:
        return None
    return from_torch(value[0]), from_torch(value[1])


def modality_arrays(value: TransformerArgs) -> ModalityArrays:
    return ModalityArrays(
        context=from_torch(value.context),
        timesteps=from_torch(value.timesteps),
        positional_embeddings=(
            from_torch(value.positional_embeddings[0]),
            from_torch(value.positional_embeddings[1]),
        ),
        context_mask=optional_array(value.context_mask),
        prompt_timestep=optional_array(value.prompt_timestep),
        self_attention_mask=optional_array(value.self_attention_mask),
        cross_positional_embeddings=optional_rope(value.cross_positional_embeddings),
        cross_scale_shift_timestep=optional_array(value.cross_scale_shift_timestep),
        cross_gate_timestep=optional_array(value.cross_gate_timestep),
    )


@mx.compile
def modulated_norm(
    x: mx.array, shift: mx.array, scale: mx.array, epsilon: float
) -> mx.array:
    return mx.fast.rms_norm(x, None, epsilon) * (1 + scale) + shift


@mx.compile
def gated_residual(x: mx.array, update: mx.array, gate: mx.array) -> mx.array:
    return x + update * gate


def attention_shape(attention: Attention) -> AttentionShape:
    return AttentionShape(
        attention.heads,
        attention.dim_head,
        attention.q_norm.eps,
        attention.rope_type == LTXRopeType.SPLIT,
    )


class MLXBlock:
    def __init__(self, block: BasicAVTransformerBlock, compiled: bool = True) -> None:
        state = cast(dict[str, torch.Tensor], block.state_dict())
        self.weights = {name: from_torch(weight) for name, weight in state.items()}
        self.shape = BlockShape(
            attention_shape(block.attn1) if hasattr(block, "attn1") else None,
            attention_shape(block.audio_attn1)
            if hasattr(block, "audio_attn1")
            else None,
            attention_shape(block.audio_to_video_attn)
            if hasattr(block, "audio_to_video_attn")
            else None,
            block.cross_attention_adaln,
            block.norm_eps,
        )
        self.compiled = compiled
        mx.eval(self.weights)

    def linear(self, x: mx.array, prefix: str) -> mx.array:
        output = x @ self.weights[prefix + ".weight"].T
        bias = self.weights.get(prefix + ".bias")
        return output if bias is None else output + bias

    def modulation(
        self, table_name: str, timestep: mx.array, begin: int, end: int
    ) -> tuple[mx.array, ...]:
        table = self.weights[table_name]
        values = (
            table[None, None, begin:end]
            + timestep.reshape(
                timestep.shape[0], timestep.shape[1], table.shape[0], -1
            )[:, :, begin:end]
        )
        return tuple(values[:, :, index] for index in range(end - begin))

    def normalize(self, x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
        if self.compiled:
            return modulated_norm(x, shift, scale, self.shape.epsilon)
        return mx.fast.rms_norm(x, None, self.shape.epsilon) * (1 + scale) + shift

    def residual(self, x: mx.array, update: mx.array, gate: mx.array) -> mx.array:
        return gated_residual(x, update, gate) if self.compiled else x + update * gate

    def rotary(
        self, x: mx.array, rope: tuple[mx.array, mx.array], shape: AttentionShape
    ) -> mx.array:
        cos, sin = rope
        if shape.split_rope:
            first, second = mx.split(x, 2, axis=-1)
            return mx.concatenate(
                [first * cos - second * sin, second * cos + first * sin], axis=-1
            )
        if cos.ndim == 3:
            cos = cos.reshape(
                x.shape[0], x.shape[2], shape.heads, shape.head_dim
            ).transpose(0, 2, 1, 3)
            sin = sin.reshape(
                x.shape[0], x.shape[2], shape.heads, shape.head_dim
            ).transpose(0, 2, 1, 3)
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated = mx.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
        return x * cos + rotated * sin

    def attention(
        self,
        x: mx.array,
        prefix: str,
        shape: AttentionShape,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        rope: tuple[mx.array, mx.array] | None = None,
        key_rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        source = x if context is None else context
        q = self.linear(x, prefix + ".to_q")
        k = self.linear(source, prefix + ".to_k")
        v = self.linear(source, prefix + ".to_v")
        q = mx.fast.rms_norm(q, self.weights[prefix + ".q_norm.weight"], shape.epsilon)
        k = mx.fast.rms_norm(k, self.weights[prefix + ".k_norm.weight"], shape.epsilon)
        batch, tokens = q.shape[:2]
        q = q.reshape(batch, tokens, shape.heads, shape.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, -1, shape.heads, shape.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, -1, shape.heads, shape.head_dim).transpose(0, 2, 1, 3)
        if rope is not None:
            q = self.rotary(q, rope, shape)
            k = self.rotary(k, rope if key_rope is None else key_rope, shape)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=shape.head_dim**-0.5, mask=mask
        )
        output = output.transpose(0, 2, 1, 3)
        if prefix + ".to_gate_logits.weight" in self.weights:
            gates = 2 * mx.sigmoid(self.linear(x, prefix + ".to_gate_logits"))
            output = output * gates[..., None]
        return self.linear(output.reshape(batch, tokens, -1), prefix + ".to_out.0")

    def self_and_text(
        self, x: mx.array, args: ModalityArrays, prefix: str, shape: AttentionShape
    ) -> mx.array:
        table = prefix + "scale_shift_table"
        shift, scale, gate = self.modulation(table, args.timesteps, 0, 3)
        normalized = self.normalize(x, shift, scale)
        update = self.attention(
            normalized,
            prefix + "attn1",
            shape,
            mask=args.self_attention_mask,
            rope=args.positional_embeddings,
        )
        x = self.residual(x, update, gate)
        normalized = mx.fast.rms_norm(x, None, self.shape.epsilon)
        context = args.context
        text_gate: mx.array | None = None
        if self.shape.cross_attention_adaln:
            shift, scale, text_gate = self.modulation(table, args.timesteps, 6, 9)
            normalized = normalized * (1 + scale) + shift
            prompt_table = self.weights[prefix + "prompt_scale_shift_table"]
            prompt = prompt_table[None, None]
            if args.prompt_timestep is not None:
                prompt = prompt + args.prompt_timestep.reshape(
                    args.prompt_timestep.shape[0], args.prompt_timestep.shape[1], 2, -1
                )
            context = context * (1 + prompt[:, :, 1]) + prompt[:, :, 0]
        update = self.attention(
            normalized, prefix + "attn2", shape, context=context, mask=args.context_mask
        )
        return x + update if text_gate is None else self.residual(x, update, text_gate)

    def cross_modulation(
        self, x: mx.array, args: ModalityArrays, table_name: str, direction: int
    ) -> tuple[mx.array, mx.array]:
        assert (
            args.cross_scale_shift_timestep is not None
            and args.cross_gate_timestep is not None
        )
        table = self.weights[table_name]
        scale_shift = table[None, None, :4] + args.cross_scale_shift_timestep.reshape(
            x.shape[0], args.cross_scale_shift_timestep.shape[1], 4, -1
        )
        scale, shift = scale_shift[:, :, direction], scale_shift[:, :, direction + 1]
        gate = table[None, None, 4] + args.cross_gate_timestep.reshape(
            x.shape[0], args.cross_gate_timestep.shape[1], -1
        )
        return self.normalize(x, shift, scale), gate

    def feed_forward(self, x: mx.array, args: ModalityArrays, prefix: str) -> mx.array:
        shift, scale, gate = self.modulation(
            prefix + "scale_shift_table", args.timesteps, 3, 6
        )
        normalized = self.normalize(x, shift, scale)
        hidden = nn.gelu_approx(self.linear(normalized, prefix + "ff.net.0.proj"))
        return self.residual(x, self.linear(hidden, prefix + "ff.net.2"), gate)

    def __call__(
        self,
        video: mx.array | None,
        audio: mx.array | None,
        video_args: ModalityArrays | None,
        audio_args: ModalityArrays | None,
    ) -> tuple[mx.array | None, mx.array | None]:
        if video is not None and video_args is not None:
            assert self.shape.video_attention is not None
            video = self.self_and_text(
                video, video_args, "", self.shape.video_attention
            )
        if audio is not None and audio_args is not None:
            assert self.shape.audio_attention is not None
            audio = self.self_and_text(
                audio, audio_args, "audio_", self.shape.audio_attention
            )
        if (
            video is not None
            and audio is not None
            and video_args is not None
            and audio_args is not None
        ):
            assert self.shape.cross_attention is not None
            video_before, audio_before = video, audio
            video_scaled, video_gate = self.cross_modulation(
                video_before, video_args, "scale_shift_table_a2v_ca_video", 0
            )
            audio_scaled, _ = self.cross_modulation(
                audio_before, audio_args, "scale_shift_table_a2v_ca_audio", 0
            )
            update = self.attention(
                video_scaled,
                "audio_to_video_attn",
                self.shape.cross_attention,
                context=audio_scaled,
                rope=video_args.cross_positional_embeddings,
                key_rope=audio_args.cross_positional_embeddings,
            )
            video = self.residual(video, update, video_gate)
            audio_scaled, audio_gate = self.cross_modulation(
                audio_before, audio_args, "scale_shift_table_a2v_ca_audio", 2
            )
            video_scaled, _ = self.cross_modulation(
                video_before, video_args, "scale_shift_table_a2v_ca_video", 2
            )
            update = self.attention(
                audio_scaled,
                "video_to_audio_attn",
                self.shape.cross_attention,
                context=video_scaled,
                rope=audio_args.cross_positional_embeddings,
                key_rope=video_args.cross_positional_embeddings,
            )
            audio = self.residual(audio, update, audio_gate)
        if video is not None and video_args is not None:
            video = self.feed_forward(video, video_args, "")
        if audio is not None and audio_args is not None:
            audio = self.feed_forward(audio, audio_args, "audio_")
        return video, audio


class MLXBlocks:
    def __init__(self, blocks: torch.nn.ModuleList) -> None:
        self.blocks = [
            MLXBlock(cast(BasicAVTransformerBlock, block)) for block in blocks
        ]

    def __call__(
        self,
        model: LTXModel,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        masks = perturbations.block_masks_cpu
        if masks is None or not bool(torch.all(masks == 1)):
            raise ValueError(
                "MLX experiment supports the distilled, unperturbed IC-LoRA path only"
            )
        if (video is not None and not video.enabled) or (
            audio is not None and not audio.enabled
        ):
            raise ValueError("Disabled modalities are outside the tested MLX path")
        video_args = None if video is None else modality_arrays(video)
        audio_args = None if audio is None else modality_arrays(audio)
        vx = None if video is None else from_torch(video.x)
        ax = None if audio is None else from_torch(audio.x)
        for index, block in enumerate(self.blocks):
            vx, ax = block(vx, ax, video_args, audio_args)
            mx.eval(vx, ax)
            if index % 8 == 7:
                mx.clear_cache()
        return (
            None
            if video is None or vx is None
            else replace(video, x=to_torch(vx, video.x)),
            None
            if audio is None or ax is None
            else replace(audio, x=to_torch(ax, audio.x)),
        )


def install_mlx_blocks(model: LTXModel) -> MLXBlocks:
    existing = getattr(model, "_mlx_backend", None)
    if existing is not None:
        return cast(MLXBlocks, existing)
    blocks = model.transformer_blocks
    blocks.to("cpu")
    backend = MLXBlocks(blocks)
    model._process_transformer_blocks = MethodType(backend, model)
    model.transformer_blocks = torch.nn.ModuleList(
        torch.nn.Identity() for _ in range(len(blocks))
    )
    model._mlx_backend = backend
    torch.mps.empty_cache()
    return backend
