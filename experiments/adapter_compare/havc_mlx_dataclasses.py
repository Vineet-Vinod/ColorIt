from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Qwen21Shape:
    heads: int
    head_dim: int
    layers: int
    axes: tuple[int, ...]
    epsilon: float
    out_channels: int
    fused_mlp: bool


@dataclass(frozen=True)
class Qwen21Segment:
    begin: int
    end: int
    causal: bool


@dataclass(frozen=True)
class Qwen21LoRA:
    down: mx.array
    up: mx.array


@dataclass(frozen=True)
class Qwen21Prefix:
    signature: tuple[object, ...]
    context: mx.array
    references: tuple[mx.array, ...]
    key_values: tuple[tuple[mx.array, mx.array], ...]
    target_rope: mx.array
