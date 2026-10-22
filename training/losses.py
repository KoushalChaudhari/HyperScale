"""Pix2Pix adversarial and reconstruction objectives."""

from __future__ import annotations

import torch
from torch import nn


def discriminator_loss(
    criterion: nn.Module,
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        criterion(real_logits, torch.ones_like(real_logits))
        + criterion(fake_logits, torch.zeros_like(fake_logits))
    )


def generator_losses(
    criterion: nn.Module,
    fake_logits: torch.Tensor,
    generated: torch.Tensor,
    target: torch.Tensor,
    lambda_l1: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adversarial = criterion(fake_logits, torch.ones_like(fake_logits))
    reconstruction = nn.functional.l1_loss(generated, target)
    return adversarial + lambda_l1 * reconstruction, adversarial, reconstruction


class VGG19ContentLoss(nn.Module):
    """Optional ImageNet VGG19 feature loss loaded only from an explicit local file."""

    def __init__(self, weights_path: str) -> None:
        super().__init__()
        from torchvision.models import vgg19

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise ValueError("VGG weights file must contain a state dictionary")
        cleaned = {key.removeprefix("module."): value for key, value in state.items()}
        network = vgg19(weights=None)
        try:
            network.load_state_dict(cleaned, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Supplied file is not a complete VGG19 state dictionary: {exc}") from exc
        self.features = nn.Sequential(*list(network.features.children())[:36]).eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        generated = (generated.clamp(-1, 1) + 1) / 2
        target = (target.clamp(-1, 1) + 1) / 2
        generated = (generated - self.mean) / self.std
        target = (target - self.mean) / self.std
        with torch.no_grad():
            target_features = self.features(target)
        return nn.functional.l1_loss(self.features(generated), target_features)
