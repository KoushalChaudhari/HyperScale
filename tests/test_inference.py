import numpy as np
import torch
from PIL import Image

from data.preprocessing import load_rgb_image
from inference.restore import load_generator, restore_image
from models import UNetGenerator


def test_inference_checkpoint_and_original_dimensions(tmp_path):
    checkpoint = tmp_path / "model.pth"
    model = UNetGenerator(base_channels=4)
    torch.save(
        {
            "generator": model.state_dict(),
            "config": {"training": {"base_channels": 4}, "data": {"image_size": 128}},
            "trained": True,
        },
        checkpoint,
    )
    loaded, device, _ = load_generator(checkpoint, "cpu")
    source = Image.fromarray(np.full((95, 151, 3), 110, dtype=np.uint8))
    result = restore_image(source, model=loaded, device=device)
    assert result.image.mode == "RGB"
    assert result.image.size == source.size
    assert result.elapsed_seconds >= 0


def test_missing_or_untrained_checkpoint_is_reported(tmp_path):
    try:
        load_generator(tmp_path / "missing.pth", "cpu")
    except FileNotFoundError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Missing checkpoint should fail")
    untrained = tmp_path / "untrained.pth"
    torch.save({"generator": UNetGenerator(base_channels=4).state_dict(), "trained": False}, untrained)
    try:
        load_generator(untrained, "cpu")
    except ValueError as exc:
        assert "trained=True" in str(exc)
    else:
        raise AssertionError("Untrained weights should not be accepted as trained")


def test_invalid_image_is_reported(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_text("not image bytes", encoding="utf-8")
    try:
        load_rgb_image(corrupt)
    except ValueError as exc:
        assert "Could not read" in str(exc)
    else:
        raise AssertionError("Corrupt image must produce a helpful error")
