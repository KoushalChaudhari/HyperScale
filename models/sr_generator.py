"""Residual PixelShuffle generator for single-image super-resolution."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


class SRGenerator(nn.Module):
    """SRGAN generator with explicit, dimension-correct X2 or X4 upsampling."""

    def __init__(self, scale: int = 4, features: int = 64, residual_blocks: int = 8) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("SRGenerator supports scales X2 and X4; X3 needs a separately designed model")
        if features < 4 or residual_blocks < 1:
            raise ValueError("features must be >= 4 and residual_blocks must be positive")
        self.scale = scale
        self.features = features
        self.residual_blocks = residual_blocks
        self.head = nn.Sequential(nn.Conv2d(3, features, 9, 1, 4), nn.PReLU(features))
        self.residual = nn.Sequential(*(ResidualBlock(features) for _ in range(residual_blocks)))
        self.residual_tail = nn.Sequential(nn.Conv2d(features, features, 3, 1, 1), nn.BatchNorm2d(features))
        upsample: list[nn.Module] = []
        for _ in range(1 if scale == 2 else 2):
            upsample.extend(
                [
                    nn.Conv2d(features, features * 4, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.PReLU(features),
                ]
            )
        self.upsample = nn.Sequential(*upsample)
        self.output = nn.Conv2d(features, 3, 9, 1, 4)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.residual_tail[0].weight)
        nn.init.zeros_(self.residual_tail[0].bias)
        # Start close to the bicubic base so an undertrained model is never garbage.
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 4 or lr.shape[1] != 3:
            raise ValueError("SRGenerator input must have shape [batch, 3, height, width]")
        features = self.head(lr)
        residual = self.residual(features)
        value = features + self.residual_tail(residual)
        residual_sr = self.output(self.upsample(value))
        base = F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return torch.clamp(base + residual_sr, -1.0, 1.0)
