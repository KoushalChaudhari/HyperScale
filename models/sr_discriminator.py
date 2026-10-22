"""Convolutional HR-image discriminator for SRGAN."""

from __future__ import annotations

import torch
from torch import nn


class SRDiscriminator(nn.Module):
    """Classify high-resolution patches using raw real/fake logits."""

    def __init__(self, features: int = 64) -> None:
        super().__init__()
        if features < 4:
            raise ValueError("features must be >= 4")
        blocks: list[nn.Module] = []
        channels = (3, features, features, features * 2, features * 2, features * 4, features * 4, features * 8, features * 8)
        for index in range(len(channels) - 1):
            stride = 1 if index % 2 == 0 else 2
            blocks.append(nn.Conv2d(channels[index], channels[index + 1], 3, stride, 1))
            if index:
                blocks.append(nn.BatchNorm2d(channels[index + 1]))
            blocks.append(nn.LeakyReLU(0.2, inplace=True))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(features * 8 * 4 * 4, features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(features * 8, 1),
        )

    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        if hr.ndim != 4 or hr.shape[1] != 3:
            raise ValueError("SRDiscriminator input must have shape [batch, 3, height, width]")
        if min(hr.shape[-2:]) < 32:
            raise ValueError("SRDiscriminator requires HR images at least 32 pixels on each side")
        return self.classifier(self.features(hr))
