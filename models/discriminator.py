"""Conditional PatchGAN discriminator."""

from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """Predict real/fake logits for patches in a degraded/clean image pair."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels * 2, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        multiplier = 1
        for next_multiplier in (2, 4, 8):
            layers.extend(
                [
                    nn.Conv2d(base_channels * multiplier, base_channels * next_multiplier, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(base_channels * next_multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            multiplier = next_multiplier
        layers.extend(
            [
                nn.Conv2d(base_channels * multiplier, base_channels * 8, 4, 1, 1, bias=False),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, degraded: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        if degraded.shape != candidate.shape or degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError("Discriminator expects matching [batch, 3, height, width] tensors")
        return self.network(torch.cat((degraded, candidate), dim=1))
