from __future__ import annotations

import json
import math
from pathlib import Path
from types import MethodType
from typing import cast

import mlx.core as mx
import numpy as np
import torch
from mlx import nn
from safetensors import safe_open

from .havc_mlx_dataclasses import Qwen21LoRA, Qwen21Prefix, Qwen21Segment, Qwen21Shape


def from_torch(value: torch.Tensor) -> mx.array:
    return mx.from_dlpack(value.detach().to("cpu").contiguous())


def to_torch(value: mx.array, reference: torch.Tensor) -> torch.Tensor:
    mx.eval(value)
    return torch.from_numpy(np.array(value.astype(mx.float32))).to(reference)


def read_lora(path: Path, strength: float = 1.0) -> dict[str, Qwen21LoRA]:
    result: dict[str, Qwen21LoRA] = {}
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        metadata = weights.metadata() or {}
        config = json.loads(metadata.get("lora_adapter_metadata", "{}"))
        scale = (
            strength
            * config.get("transformer.lora_alpha", 1)
            / config.get("transformer.r", 1)
        )
        keys = weights.keys()
        for key in keys:
            if key.endswith(".lora_A.weight"):
                name = key.removeprefix("transformer.").removesuffix(".lora_A.weight")
                # The author scales B in its stored dtype before casting at inference.
                result[name] = Qwen21LoRA(
                    from_torch(weights.get_tensor(key)),
                    from_torch(
                        weights.get_tensor(key.replace("lora_A", "lora_B")) * scale
                    ),
                )
    return result


class MLXQwen21:
    """BF16/FP32 Qwen2.1 with unmerged Viggle LoRA and segment attention."""

    def __init__(
        self,
        weights: dict[str, mx.array],
        shape: Qwen21Shape,
        lora: dict[str, Qwen21LoRA] | None = None,
    ) -> None:
        self.weights = weights
        self.shape = shape
        self.lora = lora or {}
        self.rope_cache: dict[tuple[int, ...], mx.array] = {}
        self.prefix_cache: list[Qwen21Prefix] = []
        self.prefix_cache_enabled = True
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.compiled_cached_block = mx.compile(self.cached_block)
        mx.eval(
            self.weights,
            [item.down for item in self.lora.values()],
            [item.up for item in self.lora.values()],
        )
        expected = set(self.weights)
        for name in self.lora:
            if name + ".weight" not in expected and not (
                shape.fused_mlp
                and name.endswith((".gate_layer", ".proj"))
                and name.rsplit(".", 1)[0] + ".gate_up.weight" in expected
            ):
                raise ValueError(f"LoRA target is absent from Qwen2.1: {name}")

    def branch(self, x: mx.array, name: str) -> mx.array:
        pair = self.lora[name]
        return (x @ pair.down.astype(x.dtype).T) @ pair.up.astype(x.dtype).T

    def linear(self, x: mx.array, name: str) -> mx.array:
        out = x @ self.weights[name + ".weight"].astype(x.dtype).T
        if name + ".bias" in self.weights:
            out = out + self.weights[name + ".bias"].astype(x.dtype)
        if name in self.lora:
            out = out + self.branch(x, name)
        return out

    def norm(self, x: mx.array, name: str, zero_centered: bool = False) -> mx.array:
        weight = self.weights[name + ".weight"].astype(mx.float32)
        if zero_centered:
            weight = weight + 1
        return mx.fast.rms_norm(
            x.astype(mx.float32), weight, self.shape.epsilon
        ).astype(x.dtype)

    def modulation(self, timesteps: mx.array, dtype: mx.Dtype) -> tuple[mx.array, ...]:
        t = ((timesteps * 1000).astype(dtype) / 1000).astype(dtype)
        t = mx.concatenate([t, mx.zeros((1,), dtype=dtype)])
        frequencies = mx.exp(-math.log(10000) * mx.arange(128, dtype=mx.float32) / 128)
        angles = (t.astype(mx.float32) * 1000)[:, None] * frequencies[None]
        time = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1).astype(dtype)
        time = self.linear(
            nn.silu(self.linear(time, "time_text_embed.timestep_embedder.linear_1")),
            "time_text_embed.timestep_embedder.linear_2",
        )
        values = mx.split(self.linear(nn.silu(time), "modulation.1"), 4, axis=-1)
        return time, values[0], mx.tanh(values[1]), values[2], mx.tanh(values[3])

    @staticmethod
    def token_rows(values: mx.array, prefix: int, tokens: int) -> mx.array:
        if prefix == 0:
            return values[:-1, None]
        return mx.concatenate(
            [
                mx.broadcast_to(
                    values[-1:, None], (values.shape[0] - 1, prefix, values.shape[-1])
                ),
                mx.broadcast_to(
                    values[:-1, None],
                    (values.shape[0] - 1, tokens - prefix, values.shape[-1]),
                ),
            ],
            axis=1,
        )

    def rotary(self, x: mx.array, pe: mx.array) -> mx.array:
        pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
        return (
            mx.stack(
                [
                    pe[..., 0, 0] * pairs[..., 0] + pe[..., 0, 1] * pairs[..., 1],
                    pe[..., 1, 0] * pairs[..., 0] + pe[..., 1, 1] * pairs[..., 1],
                ],
                axis=-1,
            )
            .reshape(x.shape)
            .astype(x.dtype)
        )

    def attention(
        self,
        x: mx.array,
        pe: mx.array,
        segments: list[Qwen21Segment],
        name: str,
        cached: tuple[mx.array, mx.array] | None = None,
        capture: list[tuple[mx.array, mx.array]] | None = None,
        prefix: int = 0,
    ) -> mx.array:
        batch, tokens, _ = x.shape
        q, k, v = [
            self.linear(x, name + ".to_" + key).reshape(
                batch, tokens, self.shape.heads, self.shape.head_dim
            )
            for key in ("q", "k", "v")
        ]
        q = self.rotary(self.norm(q, name + ".norm_q"), pe).transpose(0, 2, 1, 3)
        k = self.rotary(self.norm(k, name + ".norm_k"), pe).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if capture is not None:
            capture.append((mx.array(k[:, :, :prefix]), mx.array(v[:, :, :prefix])))
        if cached is not None:
            k = mx.concatenate([cached[0], k], axis=2)
            v = mx.concatenate([cached[1], v], axis=2)
            outputs = [
                mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.shape.head_dim**-0.5
                )
            ]
        else:
            outputs = [
                mx.fast.scaled_dot_product_attention(
                    q[:, :, segment.begin : segment.end],
                    k[:, :, : segment.end],
                    v[:, :, : segment.end],
                    scale=self.shape.head_dim**-0.5,
                    mask="causal" if segment.causal else None,
                )
                for segment in segments
            ]
        joined = (
            mx.concatenate(outputs, axis=2)
            .transpose(0, 2, 1, 3)
            .reshape(batch, tokens, -1)
        )
        return self.linear(joined, name + ".to_out.0")

    def block(
        self,
        x: mx.array,
        pe: mx.array,
        segments: list[Qwen21Segment],
        mod: tuple[mx.array, ...],
        prefix: int,
        index: int,
        cached: tuple[mx.array, mx.array] | None = None,
        capture: list[tuple[mx.array, mx.array]] | None = None,
    ) -> mx.array:
        name = f"transformer_blocks.{index}"
        scale1, gate1, scale2, gate2 = [
            self.token_rows(value, prefix, x.shape[1]) for value in mod
        ]
        normalized = mx.fast.layer_norm(x, None, None, self.shape.epsilon) * (
            1 + scale1
        )
        x = (
            x
            + self.attention(
                normalized, pe, segments, name + ".attn", cached, capture, prefix
            )
            * gate1
        )
        normalized = mx.fast.layer_norm(x, None, None, self.shape.epsilon) * (
            1 + scale2
        )
        mlp = name + ".img_mlp"
        if self.shape.fused_mlp:
            gu = self.linear(normalized, mlp + ".gate_up")
            if mlp + ".gate_layer" in self.lora or mlp + ".proj" in self.lora:
                if (
                    mlp + ".gate_layer" not in self.lora
                    or mlp + ".proj" not in self.lora
                ):
                    raise ValueError(
                        "Fused Viggle SwiGLU requires both gate and up LoRA"
                    )
                gu = gu + mx.concatenate(
                    [
                        self.branch(normalized, mlp + ".gate_layer"),
                        self.branch(normalized, mlp + ".proj"),
                    ],
                    axis=-1,
                )
            gate, up = mx.split(gu, 2, axis=-1)
        else:
            gate, up = (
                self.linear(normalized, mlp + ".gate_layer"),
                self.linear(normalized, mlp + ".proj"),
            )
        x = x + self.linear(nn.silu(gate) * up, mlp + ".out") * gate2
        return mx.clip(x, -65504, 65504) if x.dtype == mx.float16 else x

    def build_sequence(
        self, x: mx.array, context: mx.array, refs: list[mx.array], slots: list[int]
    ) -> tuple[mx.array, mx.array, list[Qwen21Segment]]:
        txt = self.norm(context, "txt_in.text_norm", True)
        txt = self.linear(
            nn.gelu_approx(self.linear(txt, "txt_in.in_layer")), "txt_in.out_layer"
        )
        slots = (slots + [txt.shape[1]] * len(refs))[: len(refs)]
        bounds = [0, *slots, txt.shape[1]]
        if bounds != sorted(bounds):
            raise ValueError(
                "Reference slots must be ordered and inside the text sequence"
            )
        parts: list[mx.array] = []
        ids: list[np.ndarray] = []
        segments: list[Qwen21Segment] = []
        position, length = 0, 0
        key = (
            x.shape[-2],
            x.shape[-1],
            txt.shape[1],
            *slots,
            *[size for ref in refs for size in ref.shape[-2:]],
        )
        for begin, end, img in zip(bounds[:-1], bounds[1:], [*refs, x], strict=True):
            count = end - begin
            if count:
                parts.append(txt[:, begin:end])
                ids.append(
                    np.repeat(np.arange(position, position + count)[:, None], 3, axis=1)
                )
                segments.append(Qwen21Segment(length, length + count, True))
                position += count
                length += count
            height, width = img.shape[-2:]
            parts.append(
                self.linear(
                    img.reshape(img.shape[0], img.shape[1], -1).transpose(0, 2, 1),
                    "img_in",
                )
            )
            hh = (
                np.arange(height)
                - (height - height // 2)
                + 0.5 * (height % 2 - x.shape[-2] % 2)
            )
            ww = (
                np.arange(width)
                - (width - width // 2)
                + 0.5 * (width % 2 - x.shape[-1] % 2)
            )
            ids.append(
                np.stack(
                    [
                        np.full((height, width), position),
                        np.broadcast_to(hh[:, None], (height, width)),
                        np.broadcast_to(ww[None], (height, width)),
                    ],
                    axis=-1,
                ).reshape(-1, 3)
            )
            segments.append(Qwen21Segment(length, length + height * width, False))
            position += max(height, width)
            length += height * width
        if key not in self.rope_cache:
            positions = np.concatenate(ids)
            matrices = []
            for axis, dimension in enumerate(self.shape.axes):
                omega = 1 / 10000 ** np.linspace(
                    0, (dimension - 2) / dimension, dimension // 2
                )
                angles = positions[:, axis, None] * omega[None]
                matrices.append(
                    np.stack(
                        [
                            np.cos(angles),
                            -np.sin(angles),
                            np.sin(angles),
                            np.cos(angles),
                        ],
                        axis=-1,
                    ).reshape(length, dimension // 2, 2, 2)
                )
            self.rope_cache[key] = mx.array(
                np.concatenate(matrices, axis=1).astype(np.float32)
            )[None, :, None]
        return mx.concatenate(parts, axis=1), self.rope_cache[key], segments

    def cached_block(
        self,
        x: mx.array,
        pe: mx.array,
        mod: tuple[mx.array, ...],
        index: int,
        key: mx.array,
        value: mx.array,
    ) -> mx.array:
        return self.block(x, pe, [], mod, 0, index, (key, value))

    def __call__(
        self,
        x: mx.array,
        timesteps: mx.array,
        context: mx.array,
        refs: list[mx.array] | None = None,
        slots: list[int] | None = None,
    ) -> mx.array:
        refs = refs or []
        slots = (list(slots or []) + [context.shape[1]] * len(refs))[: len(refs)]
        signature: tuple[object, ...] = (
            tuple(x.shape),
            x.dtype,
            tuple(context.shape),
            context.dtype,
            tuple((tuple(ref.shape), ref.dtype) for ref in refs),
            tuple(slots),
        )
        cached = (
            self.select_prefix(signature, context, refs)
            if self.prefix_cache_enabled
            else None
        )
        captured: list[tuple[mx.array, mx.array]] = []
        if cached is None:
            hidden, pe, segments = self.build_sequence(x, context, refs, slots)
            prefix = hidden.shape[1] - x.shape[-2] * x.shape[-1]
        else:
            hidden = self.linear(
                x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1), "img_in"
            )
            pe, segments, prefix = cached.target_rope, [], 0
        time, *mod = self.modulation(timesteps, x.dtype)
        for index in range(self.shape.layers):
            if cached is not None:
                hidden = self.compiled_cached_block(
                    hidden, pe, tuple(mod), index, *cached.key_values[index]
                )
            else:
                hidden = self.block(
                    hidden,
                    pe,
                    segments,
                    tuple(mod),
                    prefix,
                    index,
                    capture=captured if prefix and self.prefix_cache_enabled else None,
                )
            mx.eval(hidden)
        if cached is None and prefix and self.prefix_cache_enabled:
            entry = Qwen21Prefix(
                signature,
                mx.array(context),
                tuple(mx.array(ref) for ref in refs),
                tuple(captured),
                pe[:, prefix:],
            )
            mx.eval(
                entry.context, entry.references, entry.key_values, entry.target_rope
            )
            self.prefix_cache.append(entry)
            self.prefix_cache = self.prefix_cache[-2:]
        hidden = hidden[:, prefix:]
        scale = self.linear(nn.silu(time[:-1]), "norm_out.linear")[:, None]
        hidden = mx.fast.layer_norm(hidden, None, None, self.shape.epsilon) * (
            1 + scale
        )
        output = self.linear(hidden, "proj_out")
        return output.transpose(0, 2, 1).reshape(
            x.shape[0], self.shape.out_channels, *x.shape[-2:]
        )

    def select_prefix(
        self,
        signature: tuple[object, ...],
        context: mx.array,
        references: list[mx.array],
    ) -> Qwen21Prefix | None:
        for index, cached in enumerate(self.prefix_cache):
            if cached.signature != signature:
                continue
            checks = [mx.array_equal(context, cached.context)]
            checks.extend(
                mx.array_equal(ref, old)
                for ref, old in zip(references, cached.references, strict=True)
            )
            if bool(mx.all(mx.stack(checks)).item()):
                self.prefix_cache.append(self.prefix_cache.pop(index))
                self.prefix_cache_hits += 1
                return cached
        self.prefix_cache_misses += 1
        return None

    def reset_prefix_cache(self, model: torch.nn.Module, enabled: bool) -> None:
        self.prefix_cache.clear()
        self.prefix_cache_enabled = enabled

    def torch_forward(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        ref_latents: list[torch.Tensor] | None = None,
        image_slots: list[int] | None = None,
        transformer_options: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        options = transformer_options or {}
        if options.get("patches") or options.get("patches_replace"):
            raise ValueError(
                "MLX Qwen2.1 has no implementation for Comfy attention/block patches"
            )
        return to_torch(
            self(
                from_torch(x),
                from_torch(timesteps),
                from_torch(context),
                [from_torch(value) for value in ref_latents or []],
                image_slots,
            ),
            x,
        )


def install_mlx_transformer(
    model: torch.nn.Module, lora_path: Path | None = None, strength: float = 1.0
) -> MLXQwen21:
    existing = getattr(model, "_mlx_backend", None)
    if existing is not None:
        return cast(MLXQwen21, existing)
    blocks = model.get_submodule("transformer_blocks")
    first = blocks.get_submodule("0")
    attention = first.get_submodule("attn")
    shape = Qwen21Shape(
        int(attention.heads),
        int(attention.get_submodule("norm_q").get_parameter("weight").numel()),
        len(list(blocks.children())),
        tuple(model.get_submodule("pe_embedder").axes_dim),
        float(first.get_submodule("img_norm1").eps),
        int(model.out_channels),
        bool(first.get_submodule("img_mlp").fused),
    )
    model.to("cpu")
    weights = {name: from_torch(weight) for name, weight in model.state_dict().items()}
    backend = MLXQwen21(
        weights, shape, read_lora(lora_path, strength) if lora_path else None
    )
    model._forward = MethodType(backend.torch_forward, model)
    model.reset_prefix_cache = MethodType(backend.reset_prefix_cache, model)
    # Keep the native outer forward and WrapperExecutor; all numerical work is MLX.
    for name, _ in list(model.named_children()):
        setattr(model, name, torch.nn.Identity())
    model._mlx_backend = backend
    return backend
