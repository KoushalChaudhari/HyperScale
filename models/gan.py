"""Convenience wrapper for the generator and discriminator."""

from .discriminator import PatchDiscriminator
from .generator import UNetGenerator


class Pix2Pix:
    def __init__(self, base_channels: int = 32, depth: int = 7) -> None:
        self.generator = UNetGenerator(base_channels=base_channels, depth=depth)
        self.discriminator = PatchDiscriminator(base_channels=base_channels)
