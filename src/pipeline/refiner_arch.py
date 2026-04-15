from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualConvBlock(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        upsampled = self.up(x)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = nn.functional.interpolate(
                upsampled,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused = self.fuse(torch.cat([upsampled, skip], dim=1))
        return self.residual(fused)


class CostumeRefinerUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 32,
        ab_delta_scale: float = 24.0,
    ):
        super().__init__()
        self.ab_delta_scale = float(ab_delta_scale)

        self.down1 = DownBlock(input_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)

        bottleneck_channels = base_channels * 16
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            ResidualConvBlock(bottleneck_channels),
            ResidualConvBlock(bottleneck_channels),
        )

        self.up4 = UpBlock(bottleneck_channels, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        x = self.bottleneck(x)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.head(x) * self.ab_delta_scale
