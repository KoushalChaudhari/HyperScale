import torch

from models import PatchDiscriminator, UNetGenerator
from training.losses import discriminator_loss, generator_losses


def test_generator_discriminator_forward_and_backward():
    generator = UNetGenerator(base_channels=4)
    discriminator = PatchDiscriminator(base_channels=4)
    degraded = torch.randn(2, 3, 128, 128)
    target = torch.randn_like(degraded)
    generated = generator(degraded)
    assert generated.shape == degraded.shape
    logits = discriminator(degraded, generated)
    assert logits.ndim == 4 and logits.shape[1] == 1
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_g, adversarial, reconstruction = generator_losses(criterion, logits, generated, target)
    loss_d = discriminator_loss(criterion, discriminator(degraded, target), logits.detach())
    assert all(torch.isfinite(loss) and loss.ndim == 0 for loss in (loss_g, adversarial, reconstruction, loss_d))
    (loss_g + loss_d).backward()


def test_generator_requires_divisible_dimensions():
    generator = UNetGenerator(base_channels=4)
    try:
        generator(torch.randn(1, 3, 120, 128))
    except ValueError as exc:
        assert "multiples of 128" in str(exc)
    else:
        raise AssertionError("Expected an informative dimension error")
