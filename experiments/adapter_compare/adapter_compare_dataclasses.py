from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class FrameWindow:
    begin: int
    end: int
    skip: int


@dataclass(frozen=True)
class ModalityArrays:
    context: mx.array
    timesteps: mx.array
    positional_embeddings: tuple[mx.array, mx.array]
    context_mask: mx.array | None = None
    prompt_timestep: mx.array | None = None
    self_attention_mask: mx.array | None = None
    cross_positional_embeddings: tuple[mx.array, mx.array] | None = None
    cross_scale_shift_timestep: mx.array | None = None
    cross_gate_timestep: mx.array | None = None


@dataclass(frozen=True)
class AttentionShape:
    heads: int
    head_dim: int
    epsilon: float
    split_rope: bool


@dataclass(frozen=True)
class BlockShape:
    video_attention: AttentionShape | None
    audio_attention: AttentionShape | None
    cross_attention: AttentionShape | None
    cross_attention_adaln: bool
    epsilon: float
