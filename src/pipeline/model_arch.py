from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
import warnings

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torchvision import models


warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.",
    category=FutureWarning,
)


class NormType(Enum):
    Batch = "batch"
    BatchZero = "batch_zero"
    Weight = "weight"
    Spectral = "spectral"


def relu(inplace: bool = False, leaky: float | None = None) -> nn.Module:
    if leaky is not None:
        return nn.LeakyReLU(inplace=inplace, negative_slope=leaky)
    return nn.ReLU(inplace=inplace)


def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0.0 if norm_type == NormType.BatchZero else 1.0)
    return bn


class SelfAttention(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.query = spectral_norm(nn.Conv1d(n_channels, n_channels // 8, 1, bias=False))
        self.key = spectral_norm(nn.Conv1d(n_channels, n_channels // 8, 1, bias=False))
        self.value = spectral_norm(nn.Conv1d(n_channels, n_channels, 1, bias=False))
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor) -> Tensor:
        size = x.size()
        flat = x.view(*size[:2], -1)
        query = self.query(flat)
        key = self.key(flat)
        value = self.value(flat)
        beta = F.softmax(torch.bmm(query.permute(0, 2, 1).contiguous(), key), dim=1)
        out = self.gamma * torch.bmm(value, beta) + flat
        return out.view(*size).contiguous()


def init_default(module: nn.Module, init_fn) -> nn.Module:
    init_fn(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0.0)
    return module


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int | None = None,
    bias: bool | None = None,
    norm_type: NormType = NormType.Batch,
    use_activ: bool = True,
    leaky: float | None = None,
    transpose: bool = False,
    init=nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
) -> nn.Sequential:
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn
    if bias is None:
        bias = not bn

    conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
    conv = init_default(
        conv_cls(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    elif norm_type == NormType.Weight:
        conv = weight_norm(conv)

    layers: list[nn.Module] = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def icnr(weight: Tensor, scale: int = 2, init=nn.init.kaiming_normal_) -> None:
    ni, nf, height, width = weight.shape
    ni2 = int(ni / (scale**2))
    kernel = init(torch.zeros([ni2, nf, height, width])).transpose(0, 1)
    kernel = kernel.contiguous().view(ni2, nf, -1)
    kernel = kernel.repeat(1, 1, scale**2)
    kernel = kernel.contiguous().view([nf, ni, height, width]).transpose(0, 1)
    weight.data.copy_(kernel)


class SequentialEx(nn.Module):
    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        result = x
        for layer in self.layers:
            result.orig = x
            next_result = layer(result)
            result.orig = None
            result = next_result
        return result

    def __getitem__(self, index: int) -> nn.Module:
        return self.layers[index]


class MergeLayer(nn.Module):
    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense

    def forward(self, x: Tensor) -> Tensor:
        if self.dense:
            return torch.cat([x, x.orig], dim=1)
        return x + x.orig


class SigmoidRange(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * (self.high - self.low) + self.low


def res_block(
    nf: int,
    dense: bool = False,
    norm_type: NormType = NormType.Batch,
    bottle: bool = False,
    **conv_kwargs,
) -> SequentialEx:
    norm2 = norm_type
    if not dense and norm_type == NormType.Batch:
        norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(
        custom_conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
        custom_conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
        MergeLayer(dense),
    )


class Hook:
    def __init__(self, module: nn.Module):
        self.stored = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, inputs, output) -> None:
        self.stored = output.detach()

    def remove(self) -> None:
        self.hook.remove()


class HookList:
    def __init__(self, modules: Iterable[nn.Module]):
        self.hooks = [Hook(module) for module in modules]

    def __getitem__(self, index: int) -> Hook:
        return self.hooks[index]

    def __iter__(self):
        return iter(self.hooks)

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()


def model_sizes(model: nn.Module, size: tuple[int, int] = (256, 256)) -> list[torch.Size]:
    hooks = HookList(model)
    try:
        _ = dummy_eval(model, size)
        return [hook.stored.shape for hook in hooks]
    finally:
        hooks.remove()


def dummy_eval(model: nn.Module, size: tuple[int, int]) -> Tensor:
    model.eval()
    dummy = torch.zeros(1, in_channels(model), *size)
    with torch.no_grad():
        return model(dummy)


def in_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
            return child.in_channels
    raise ValueError("Unable to determine model input channels.")


def create_resnet101_body() -> nn.Sequential:
    backbone = models.resnet101(weights=None)
    return nn.Sequential(*list(backbone.children())[:-2])


def _get_sfs_idxs(sizes: list[torch.Size]) -> list[int]:
    feature_sizes = [shape[-1] for shape in sizes]
    sfs_idxs = list(np.where(np.array(feature_sizes[:-1]) != np.array(feature_sizes[1:]))[0])
    if feature_sizes[0] != feature_sizes[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


class CustomPixelShuffleICNR(nn.Module):
    def __init__(
        self,
        ni: int,
        nf: int | None = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float | None = None,
        **kwargs,
    ):
        super().__init__()
        nf = ni if nf is None else nf
        self.conv = custom_conv_layer(ni, nf * (scale**2), ks=1, use_activ=False, **kwargs)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)
        self.do_blur = blur

    def forward(self, x: Tensor) -> Tensor:
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x


class UnetBlockWide(nn.Module):
    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        n_out: int,
        hook: Hook,
        blur: bool = False,
        leaky: float | None = None,
        self_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hook = hook
        up_out = n_out // 2
        x_out = n_out // 2
        self.shuf = CustomPixelShuffleICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        self.conv = custom_conv_layer(
            up_out + x_in_c,
            x_out,
            leaky=leaky,
            self_attention=self_attention,
            **kwargs,
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        skip = self.hook.stored
        up_out = self.shuf(up_in)
        if skip.shape[-2:] != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, skip.shape[-2:], mode="nearest")
        cat = self.relu(torch.cat([up_out, self.bn(skip)], dim=1))
        return self.conv(cat)


class DeoldifyVideoModel(SequentialEx):
    def __init__(self, nf_factor: int = 2):
        encoder = create_resnet101_body()
        extra_bn = True
        imsize = (256, 256)
        sfs_sizes = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_sizes)))
        sfs = HookList([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_sizes[-1][1]
        middle_conv = nn.Sequential(
            custom_conv_layer(
                ni, ni * 2, norm_type=NormType.Spectral, extra_bn=extra_bn
            ),
            custom_conv_layer(
                ni * 2, ni, norm_type=NormType.Spectral, extra_bn=extra_bn
            ),
        ).eval()
        x = middle_conv(x)
        layers: list[nn.Module] = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        nf = 512 * nf_factor
        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c = int(x.shape[1])
            x_in_c = int(sfs_sizes[idx][1])
            sa = i == len(sfs_idxs) - 3
            n_out = nf if not_final else nf // 2
            unet_block = UnetBlockWide(
                up_in_c,
                x_in_c,
                n_out,
                sfs[i],
                blur=True,
                self_attention=sa,
                norm_type=NormType.Spectral,
                extra_bn=extra_bn,
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = int(x.shape[1])
        if imsize != sfs_sizes[0][-2:]:
            layers.append(PixelShuffleICNR(ni))
        layers.append(MergeLayer(dense=True))
        ni += in_channels(encoder)
        layers.append(res_block(ni, bottle=False, norm_type=NormType.Spectral))
        layers.append(
            custom_conv_layer(
                ni,
                3,
                ks=1,
                use_activ=False,
                norm_type=NormType.Spectral,
            )
        )
        layers.append(SigmoidRange(-3.0, 3.0))
        super().__init__(*layers)
        self.sfs = sfs

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()


class PixelShuffleICNR(nn.Module):
    def __init__(
        self,
        ni: int,
        nf: int | None = None,
        scale: int = 2,
        blur: bool = False,
        norm_type: NormType = NormType.Weight,
        leaky: float | None = None,
    ):
        super().__init__()
        nf = ni if nf is None else nf
        self.conv = custom_conv_layer(ni, nf * (scale**2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)
        self.do_blur = blur

    def forward(self, x: Tensor) -> Tensor:
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.do_blur else x
