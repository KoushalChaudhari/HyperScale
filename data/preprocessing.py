"""Pillow and tensor conversion helpers shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def load_rgb_image(source: str | Path | Image.Image) -> Image.Image:
    """Open an image safely and normalize grayscale/RGBA inputs to RGB."""
    try:
        if isinstance(source, (str, Path)):
            with Image.open(source) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read a supported image: {exc}") from exc


def prepare_training_image(image: Image.Image, size: int) -> Image.Image:
    """Resize and center-crop an image to the square training dimensions."""
    image = ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    return image


def normalize_image(image: Image.Image | np.ndarray) -> torch.Tensor:
    """Convert RGB pixels in [0, 255] to a CHW float tensor in [-1, 1]."""
    array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape [height, width, 3]")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).div_(127.5).sub_(1.0)


def denormalize_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [-1, 1] CHW or single-item BCHW tensor to an RGB Pillow image."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("A batch must contain exactly one image")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected a [3, height, width] tensor")
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy(), mode="RGB")
