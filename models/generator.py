"""U-Net generator used by the conditional restoration GAN."""

from __future__ import annotations

import torch
from torch import nn


class UNetGenerator(nn.Module):
    """Pix2Pix-style U-Net mapping a degraded RGB image to restored RGB."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if depth < 2 or base_channels < 1:
            raise ValueError("depth must be >= 2 and base_channels must be positive")

        channels = [min(base_channels * (2**i), base_channels * 8) for i in range(depth)]
        self.required_multiple = 2**depth
        self.encoder = nn.ModuleList()
        previous = in_channels
        for index, current in enumerate(channels):
            layers: list[nn.Module] = [nn.Conv2d(previous, current, 4, 2, 1, bias=index == 0)]
            if index:
                layers.append(nn.BatchNorm2d(current))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.encoder.append(nn.Sequential(*layers))
            previous = current

        self.decoder = nn.ModuleList()
        for index, skip_channels in enumerate(reversed(channels[:-1])):
            layers = [
                nn.ConvTranspose2d(previous, skip_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(skip_channels),
                nn.ReLU(inplace=True),
            ]
            if index < 3 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.decoder.append(nn.Sequential(*layers))
            previous = skip_channels * 2
        self.output = nn.Sequential(
            nn.ConvTranspose2d(previous, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Generator input must have shape [batch, 3, height, width]")
        multiple = self.required_multiple
        if image.shape[-2] % multiple or image.shape[-1] % multiple:
            raise ValueError(f"Input height and width must be multiples of {multiple}")
        skips: list[torch.Tensor] = []
        value = image
        for layer in self.encoder:
            value = layer(value)
            skips.append(value)
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            value = layer(value)
            if value.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError("U-Net skip dimensions do not match")
            value = torch.cat((value, skip), dim=1)
        return self.output(value)
