"""SRGAN model bundle kept separate from data and training orchestration."""

from __future__ import annotations

import torch

from models.sr_discriminator import SRDiscriminator
from models.sr_generator import SRGenerator


class SRGAN(torch.nn.Module):
    def __init__(self, scale: int = 4, features: int = 64, residual_blocks: int = 8) -> None:
        super().__init__()
        self.generator = SRGenerator(scale, features, residual_blocks)
        self.discriminator = SRDiscriminator(features)

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.generator(lr)
