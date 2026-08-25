"""MLX inference implementation of the official DeepRemaster networks.

The original implementation is available from
https://github.com/satoshiiizuka/siggraphasia2019_remastering.  It stores
volumes as ``NCTHW``; this module deliberately uses MLX's native ``NTHWC``
layout throughout.  The checkpoint converter below folds every evaluation
BatchNorm3d into the preceding convolution, so the runtime contains only
Conv3d, ELU, interpolation, and attention operations.

This module is intentionally self-contained.  Importing it does not require
MLX so installations which use the PyTorch path remain usable; constructing a
model or converting weights does require MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # MLX is optional until the MLX backend is selected.
    import mlx.core as mx
    import mlx.nn as nn

    _MLX_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised on non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR = exc


_Module = nn.Module if nn is not None else object

_BN_EPS = 1e-5  # torch.nn.BatchNorm3d default, used by the official network.
_LUMA_MEAN = 0.4462414
_REFERENCE_MEAN = 0.48


def mlx_available() -> bool:
    """Return whether MLX could be imported in this Python environment."""

    return mx is not None


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise RuntimeError(
            "The DeepRemaster MLX backend requires the optional 'mlx' package. "
            "Install it on Apple silicon before selecting this backend."
        ) from _MLX_IMPORT_ERROR


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(value)
    if len(result) != 3:
        raise ValueError(f"Expected three dimensions, got {value!r}")
    return result  # type: ignore[return-value]


def _require_volume(name: str, x: Any, channels: int) -> None:
    if x.ndim != 5 or x.shape[-1] != channels:
        raise ValueError(
            f"{name} must have NTHWC shape with {channels} channels; got {x.shape}"
        )


class _Conv3d(_Module):
    """A channels-last 3-D convolution with explicitly named MLX weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = (1, 3, 3),
        stride: int | Sequence[int] = (1, 1, 1),
        padding: int | Sequence[int] = (0, 1, 1),
    ) -> None:
        _require_mlx()
        super().__init__()
        kernel = _triple(kernel_size)
        # MLX Conv3d weights are ODHWI, unlike PyTorch's OIDHW.
        self.weight = mx.zeros((out_channels, *kernel, in_channels))
        self.bias = mx.zeros((out_channels,))
        self.stride = _triple(stride)
        self.padding = _triple(padding)

    def __call__(self, x: Any) -> Any:
        return mx.conv3d(x, self.weight, stride=self.stride, padding=self.padding) + self.bias


class _TempConv(_Module):
    """Official TempConv after folding its BatchNorm3d into ``conv``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = (1, 3, 3),
        stride: int | Sequence[int] = (1, 1, 1),
        padding: int | Sequence[int] = (0, 1, 1),
    ) -> None:
        _require_mlx()
        super().__init__()
        self.conv = _Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def __call__(self, x: Any) -> Any:
        return nn.elu(self.conv(x))


class _UpsampleConv(_Module):
    """Trilinear NTHWC upsample followed by a fused Conv3d + ELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=(1, 2, 2), mode="linear", align_corners=False
        )
        self.conv = _Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def __call__(self, x: Any) -> Any:
        return nn.elu(self.conv(self.upsample(x)))


class _UpsampleConcat(_Module):
    """Official UpsampleConcat with its single fused TempConv."""

    def __init__(self, in_channels_up: int, in_channels_flat: int, out_channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=(1, 2, 2), mode="linear", align_corners=False
        )
        self.conv = _Conv3d(
            in_channels_up + in_channels_flat,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )

    def __call__(self, x_up: Any, x_flat: Any) -> Any:
        return nn.elu(self.conv(mx.concatenate((self.upsample(x_up), x_flat), axis=-1)))


class _EdgePad3d(_Module):
    """PyTorch ReplicationPad3d in NTHWC order."""

    def __init__(self, pad_t: int, pad_h: int, pad_w: int) -> None:
        _require_mlx()
        super().__init__()
        self.pad_width = ((0, 0), (pad_t, pad_t), (pad_h, pad_h), (pad_w, pad_w), (0, 0))

    def __call__(self, x: Any) -> Any:
        return mx.pad(x, self.pad_width, mode="edge")


class SourceReferenceAttention(_Module):
    """Source-reference attention with an exact memory-bounded path.

    The tiled implementation performs the same unscaled softmax(q @ k.T) as
    the official code.  It uses online log-sum-exp accumulation over reference
    tiles, avoiding the otherwise enormous ``Ns x Nr`` score allocation.
    """

    def __init__(
        self,
        source_channels: int = 512,
        reference_channels: int = 512,
        *,
        dense_max_scores: int = 32_000_000,
        source_tile_size: int = 1024,
        reference_tile_size: int = 2048,
    ) -> None:
        _require_mlx()
        super().__init__()
        if dense_max_scores < 1 or source_tile_size < 1 or reference_tile_size < 1:
            raise ValueError("Attention limits must be positive")
        self.query = _Conv3d(source_channels, source_channels // 8, kernel_size=1, padding=0)
        self.key = _Conv3d(reference_channels, reference_channels // 8, kernel_size=1, padding=0)
        self.value = _Conv3d(reference_channels, reference_channels, kernel_size=1, padding=0)
        self.gamma = mx.zeros((1,))
        self.dense_max_scores = dense_max_scores
        self.source_tile_size = source_tile_size
        self.reference_tile_size = reference_tile_size

    @staticmethod
    def _to_tokens(x: Any) -> Any:
        # NTHWC -> N, (T*H*W), C; MLX reshape preserves the desired order.
        return x.reshape((x.shape[0], -1, x.shape[-1]))

    def _dense(self, query: Any, key: Any, value: Any) -> Any:
        scores = mx.matmul(query, mx.swapaxes(key, -1, -2))
        attention = mx.softmax(scores, axis=-1)
        return mx.matmul(attention, value)

    def _tiled(self, query: Any, key: Any, value: Any) -> Any:
        """Compute softmax(QK^T)V with stable, exact reference tiling."""

        batch, source_positions, _ = query.shape
        reference_positions = key.shape[1]
        key_t = mx.swapaxes(key, -1, -2)
        outputs = []

        for source_start in range(0, source_positions, self.source_tile_size):
            source_end = min(source_start + self.source_tile_size, source_positions)
            query_tile = query[:, source_start:source_end, :]
            tile_positions = source_end - source_start

            # The first reference tile initializes each online-softmax state.
            running_max = None
            running_sum = None
            running_value = None
            for reference_start in range(0, reference_positions, self.reference_tile_size):
                reference_end = min(reference_start + self.reference_tile_size, reference_positions)
                scores = mx.matmul(query_tile, key_t[:, :, reference_start:reference_end])
                tile_max = mx.max(scores, axis=-1)

                if running_max is None:
                    probabilities = mx.exp(scores - tile_max[..., None])
                    running_max = tile_max
                    running_sum = mx.sum(probabilities, axis=-1)
                    running_value = mx.matmul(
                        probabilities, value[:, reference_start:reference_end, :]
                    )
                    continue

                new_max = mx.maximum(running_max, tile_max)
                old_scale = mx.exp(running_max - new_max)
                probabilities = mx.exp(scores - new_max[..., None])
                running_sum = running_sum * old_scale + mx.sum(probabilities, axis=-1)
                running_value = (
                    running_value * old_scale[..., None]
                    + mx.matmul(probabilities, value[:, reference_start:reference_end, :])
                )
                running_max = new_max

            # reference_positions is always positive for real reference/self attention.
            if running_value is None or running_sum is None:  # pragma: no cover
                raise ValueError("Attention needs at least one reference position")
            outputs.append(running_value / running_sum[..., None])

        return mx.concatenate(outputs, axis=1)

    def __call__(self, source: Any, reference: Any) -> Any:
        _require_volume("source", source, self.query.weight.shape[-1])
        _require_volume("reference", reference, self.key.weight.shape[-1])
        if source.shape[0] != reference.shape[0]:
            raise ValueError("Source and reference attention batches must match")

        source_shape = source.shape
        query = self._to_tokens(self.query(source))
        key = self._to_tokens(self.key(reference))
        value = self._to_tokens(self.value(reference))
        score_count = query.shape[0] * query.shape[1] * key.shape[1]
        if score_count <= self.dense_max_scores:
            attended = self._dense(query, key, value)
        else:
            attended = self._tiled(query, key, value)
        return source + self.gamma.reshape((1, 1, 1, 1, 1)) * attended.reshape(source_shape)


class DeepRemasterRestoration(_Module):
    """The official ``NetworkR`` in NTHWC format."""

    def __init__(self) -> None:
        _require_mlx()
        super().__init__()
        self.layers = [
            _EdgePad3d(1, 1, 1),
            _TempConv(1, 64, (3, 3, 3), (1, 2, 2), (0, 0, 0)),
            _TempConv(64, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _TempConv(128, 128, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _TempConv(128, 256, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            _TempConv(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _TempConv(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _TempConv(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _TempConv(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _UpsampleConv(256, 128),
            _TempConv(128, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _TempConv(64, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            _UpsampleConv(64, 16),
            _Conv3d(16, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        ]

    def __call__(self, x: Any) -> Any:
        _require_volume("luma", x, 1)
        # MLX accepts mixed dtypes, but feeding FP32 pixels to an FP16 model
        # promotes the convolution path and defeats the compact checkpoint.
        x = x.astype(self.layers[1].conv.weight.dtype)
        out = x - _LUMA_MEAN
        for layer in self.layers:
            out = layer(out)
        return mx.clip(x + mx.tanh(out), 0.0, 1.0)


@dataclass(frozen=True)
class ReferenceFeatures:
    """Reference encoder outputs, reusable for every source five-frame block."""

    level8: Any
    level16: Any


class DeepRemasterColorization(_Module):
    """The official ``NetworkC`` with source-reference cache support."""

    def __init__(
        self,
        *,
        dense_max_scores: int = 32_000_000,
        source_tile_size: int = 1024,
        reference_tile_size: int = 2048,
    ) -> None:
        _require_mlx()
        super().__init__()
        attention_kwargs = dict(
            dense_max_scores=dense_max_scores,
            source_tile_size=source_tile_size,
            reference_tile_size=reference_tile_size,
        )
        self.down1 = [
            _EdgePad3d(0, 1, 1),
            _TempConv(1, 64, stride=(1, 2, 2), padding=(0, 0, 0)),
            _TempConv(64, 128),
            _TempConv(128, 128),
            _TempConv(128, 256, stride=(1, 2, 2)),
            _TempConv(256, 256),
            _TempConv(256, 256),
            _TempConv(256, 512, stride=(1, 2, 2)),
            _TempConv(512, 512),
            _TempConv(512, 512),
        ]
        self.flat = [_TempConv(512, 512), _TempConv(512, 512)]
        self.down2 = [_TempConv(512, 512, stride=(1, 2, 2)), _TempConv(512, 512)]
        self.stattn1 = SourceReferenceAttention(**attention_kwargs)
        self.stattn2 = SourceReferenceAttention(**attention_kwargs)
        self.selfattn1 = SourceReferenceAttention(**attention_kwargs)
        self.conv1 = _TempConv(512, 512)
        self.up1 = _UpsampleConcat(512, 512, 512)
        self.selfattn2 = SourceReferenceAttention(**attention_kwargs)
        self.conv2 = _TempConv(512, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.up2 = [_UpsampleConv(256, 128), _TempConv(128, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1))]
        self.up3 = [_UpsampleConv(64, 32), _TempConv(32, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1))]
        self.up4 = [_UpsampleConv(16, 8), _Conv3d(8, 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))]
        self.reffeatnet1 = [
            _TempConv(3, 64, stride=(1, 2, 2)),
            _TempConv(64, 128),
            _TempConv(128, 128),
            _TempConv(128, 256, stride=(1, 2, 2)),
            _TempConv(256, 256),
            _TempConv(256, 256),
            _TempConv(256, 512, stride=(1, 2, 2)),
            _TempConv(512, 512),
            _TempConv(512, 512),
        ]
        self.reffeatnet2 = [_TempConv(512, 512, stride=(1, 2, 2)), _TempConv(512, 512), _TempConv(512, 512)]

    @staticmethod
    def _run(layers: Sequence[Any], x: Any) -> Any:
        for layer in layers:
            x = layer(x)
        return x

    def encode_references(self, references: Any) -> ReferenceFeatures:
        """Encode N,R,H,W,3 references once for all source blocks in a clip."""

        _require_volume("references", references, 3)
        references = references.astype(self.reffeatnet1[0].conv.weight.dtype)
        level8 = self._run(self.reffeatnet1, references - _REFERENCE_MEAN)
        level16 = self._run(self.reffeatnet2, level8)
        return ReferenceFeatures(level8=level8, level16=level16)

    def __call__(
        self,
        luma: Any,
        references: Any | None = None,
        *,
        reference_features: ReferenceFeatures | None = None,
    ) -> Any:
        """Colorize luma blocks; pass cached features for the high-throughput path."""

        _require_volume("luma", luma, 1)
        luma = luma.astype(self.down1[1].conv.weight.dtype)
        if references is not None and reference_features is not None:
            raise ValueError("Pass either references or reference_features, not both")
        if references is not None:
            reference_features = self.encode_references(references)

        x1 = self._run(self.down1, luma - _LUMA_MEAN)
        if reference_features is not None:
            x1 = self.stattn1(x1, reference_features.level8)
        x2 = self._run(self.flat, x1)
        out = self._run(self.down2, x1)
        if reference_features is not None:
            out = self.stattn2(out, reference_features.level16)
        out = self.conv1(out)
        out = self.selfattn1(out, out)
        out = self.up1(out, x2)
        out = self.selfattn2(out, out)
        out = self.conv2(out)
        out = self._run(self.up2, out)
        out = self._run(self.up3, out)
        out = self._run(self.up4, out)
        return mx.sigmoid(out)


class DeepRemasterMLX(_Module):
    """Combined restoration/colorization API matching the official model flow."""

    def __init__(self, **attention_kwargs: Any) -> None:
        _require_mlx()
        super().__init__()
        self.restoration = DeepRemasterRestoration()
        self.colorization = DeepRemasterColorization(**attention_kwargs)

    def prepare_references(self, references: Any) -> ReferenceFeatures:
        return self.colorization.encode_references(references)

    def __call__(
        self,
        luma: Any,
        references: Any | None = None,
        *,
        reference_features: ReferenceFeatures | None = None,
        colorize: bool = True,
    ) -> tuple[Any, Any | None]:
        restored_luma = self.restoration(luma)
        if not colorize:
            return restored_luma, None
        return restored_luma, self.colorization(
            restored_luma, references, reference_features=reference_features
        )


def _torch_load_state_dict(path: Path) -> Mapping[str, Mapping[str, Any]]:
    """Load only tensors from a trusted-format PyTorch checkpoint.

    ``weights_only=True`` prevents pickle object construction.  It requires
    PyTorch 2.6+, which this project already pins.  The key/shape validation in
    the converter is deliberately strict so a different checkpoint cannot be
    silently interpreted as DeepRemaster weights.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project dependency.
        raise RuntimeError("Converting DeepRemaster checkpoints requires PyTorch") from exc
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - old Torch is intentionally unsupported.
        raise RuntimeError("DeepRemaster conversion requires PyTorch >= 2.6 for safe loading") from exc
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"modelR", "modelC"}:
        raise ValueError("Expected the official checkpoint with exactly modelR and modelC")
    for name in ("modelR", "modelC"):
        if not isinstance(checkpoint[name], Mapping):
            raise ValueError(f"Checkpoint {name} is not a state dictionary")
    return checkpoint


def _convert_tensor(tensor: Any, dtype: str) -> Any:
    """Move a torch tensor to an MLX array, transposing Conv3d when needed."""

    if tensor.ndim == 5:
        tensor = tensor.permute(0, 2, 3, 4, 1).contiguous()
    array = mx.array(tensor.numpy())
    if dtype == "float16":
        return array.astype(mx.float16)
    if dtype == "float32":
        return array.astype(mx.float32)
    raise ValueError("dtype must be 'float32' or 'float16'")


def _convert_state_dict(state_dict: Mapping[str, Any], dtype: str) -> dict[str, Any]:
    """Map one official network state dictionary to fused MLX parameter names."""

    import torch

    converted: dict[str, Any] = {}
    consumed: set[str] = set()
    for key, tensor in state_dict.items():
        if key.endswith(".bn.num_batches_tracked"):
            consumed.add(key)
            continue
        if key.endswith(".conv3d.weight"):
            prefix = key[: -len("conv3d.weight")]
            bias_key = prefix + "conv3d.bias"
            bn_prefix = prefix + "bn."
            bn_keys = [
                bn_prefix + "weight",
                bn_prefix + "bias",
                bn_prefix + "running_mean",
                bn_prefix + "running_var",
            ]
            required = [bias_key, *bn_keys]
            if not all(candidate in state_dict for candidate in required):
                raise ValueError(f"Missing Conv3d/BatchNorm tensor(s) for {prefix}")
            # This is the exact evaluation-time BatchNorm3d fusion.
            scale = state_dict[bn_keys[0]] / torch.sqrt(state_dict[bn_keys[3]] + _BN_EPS)
            fused_weight = tensor * scale[:, None, None, None, None]
            fused_bias = (state_dict[bias_key] - state_dict[bn_keys[2]]) * scale + state_dict[bn_keys[1]]
            # UpsampleConcat owns a TempConv as ``conv3d`` in PyTorch, while
            # the MLX implementation exposes that fused convolution directly
            # as ``up1.conv``.  Other fused blocks keep their original path.
            mlx_prefix = prefix.replace(".conv3d.", ".")
            converted[mlx_prefix + "conv.weight"] = _convert_tensor(fused_weight, dtype)
            converted[mlx_prefix + "conv.bias"] = _convert_tensor(fused_bias, dtype)
            consumed.update({key, *required})
            continue
        if ".bn." in key:
            # It was consumed together with its preceding convolution.
            if key not in consumed:
                raise ValueError(f"Unexpected standalone BatchNorm tensor {key}")
            continue
        if key.endswith(".conv3d.bias"):
            if key not in consumed:
                raise ValueError(f"Unexpected standalone Conv3d bias {key}")
            continue

        mapped = (
            key.replace(".query_conv.", ".query.")
            .replace(".key_conv.", ".key.")
            .replace(".value_conv.", ".value.")
        )
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Checkpoint value {key} is not a tensor")
        converted[mapped] = _convert_tensor(tensor, dtype)
        consumed.add(key)

    unexpected = set(state_dict) - consumed
    if unexpected:
        raise ValueError(f"Unconverted DeepRemaster weights: {sorted(unexpected)}")
    return converted


def convert_official_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    dtype: str = "float32",
) -> dict[str, int | str]:
    """Safely convert official ``remasternet.pth.tar`` to MLX safetensors.

    The result has ``restoration.`` and ``colorization.`` parameter prefixes
    and loads strictly with ``DeepRemasterMLX().load_weights(output_path)``.
    ``float32`` is the parity default; use ``float16`` only after accepting the
    measured image-error tradeoff.
    """

    _require_mlx()
    source = Path(checkpoint_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.suffix != ".safetensors":
        raise ValueError("MLX checkpoint output must use the .safetensors extension")
    if dtype not in {"float32", "float16"}:
        raise ValueError("dtype must be 'float32' or 'float16'")
    destination.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = _torch_load_state_dict(source)
    restoration = _convert_state_dict(checkpoint["modelR"], dtype)
    colorization = _convert_state_dict(checkpoint["modelC"], dtype)
    weights = {
        **{f"restoration.{name}": value for name, value in restoration.items()},
        **{f"colorization.{name}": value for name, value in colorization.items()},
    }
    # Evaluate before serializing so errors surface before an apparently valid file.
    mx.eval(*weights.values())
    mx.save_safetensors(
        str(destination),
        weights,
        metadata={
            "format": "deepremaster-mlx-fused-eval-bn-v1",
            "source": source.name,
            "dtype": dtype,
        },
    )
    return {"path": str(destination), "dtype": dtype, "tensors": len(weights)}


def load_deepremaster_mlx(
    weight_path: str | Path,
    **attention_kwargs: Any,
) -> DeepRemasterMLX:
    """Create and strictly load a fused DeepRemaster MLX model."""

    _require_mlx()
    model = DeepRemasterMLX(**attention_kwargs)
    model.load_weights(str(weight_path), strict=True)
    model.eval()
    return model


__all__ = [
    "DeepRemasterColorization",
    "DeepRemasterMLX",
    "DeepRemasterRestoration",
    "ReferenceFeatures",
    "SourceReferenceAttention",
    "convert_official_checkpoint",
    "load_deepremaster_mlx",
    "mlx_available",
]
