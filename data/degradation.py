"""Randomized, configurable synthetic degradations for paired training."""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_PROBABILITIES = {
    "gaussian_noise": 0.8,
    "salt_pepper": 0.15,
    "scratches": 0.3,
    "damage_masks": 0.12,
    "blur": 0.25,
    "jpeg": 0.25,
    "brightness_contrast": 0.25,
}


def degrade_image(
    image: Image.Image,
    rng: random.Random | None = None,
    probabilities: dict[str, float] | None = None,
) -> Image.Image:
    """Apply a randomized subset of mild, aligned image corruptions."""
    rng = rng or random.Random()
    if probabilities is None:
        probabilities = DEFAULT_PROBABILITIES
    else:
        unknown = probabilities.keys() - DEFAULT_PROBABILITIES.keys()
        if unknown:
            raise ValueError(f"Unknown degradation operations: {sorted(unknown)}")
        if any(not 0 <= value <= 1 for value in probabilities.values()):
            raise ValueError("Degradation probabilities must be between 0 and 1")
        probabilities = {**DEFAULT_PROBABILITIES, **probabilities}
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    if rng.random() < probabilities.get("gaussian_noise", 0):
        sigma = rng.uniform(3.0, 15.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, rgb.shape)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < probabilities.get("salt_pepper", 0):
        mask = np.random.default_rng(rng.randrange(2**32)).random(rgb.shape[:2])
        rgb[mask < 0.0015] = 0
        rgb[mask > 0.9985] = 255
    if rng.random() < probabilities.get("scratches", 0):
        scratch_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(scratch_img)
        width, height = scratch_img.size
        for _ in range(rng.randint(1, 5)):
            start = (rng.randrange(width), rng.randrange(height))
            end = (rng.randrange(width), rng.randrange(height))
            color = rng.choice([(20, 20, 20), (235, 225, 205), (255, 255, 255)])
            draw.line((start, end), fill=color, width=rng.randint(1, 2))
        rgb = np.asarray(scratch_img).copy()
    if rng.random() < probabilities.get("damage_masks", 0):
        height, width = rgb.shape[:2]
        for _ in range(rng.randint(1, 2)):
            box_w = rng.randint(max(2, width // 24), max(3, width // 8))
            box_h = rng.randint(max(2, height // 24), max(3, height // 8))
            x, y = rng.randrange(max(1, width - box_w)), rng.randrange(max(1, height - box_h))
            cv2.rectangle(rgb, (x, y), (x + box_w, y + box_h), (110, 100, 90), -1)
    if rng.random() < probabilities.get("blur", 0):
        kernel = rng.choice([3, 5])
        rgb = cv2.GaussianBlur(rgb, (kernel, kernel), rng.uniform(0.1, 1.0))
    result = Image.fromarray(rgb, mode="RGB")
    if rng.random() < probabilities.get("jpeg", 0):
        import io

        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=rng.randint(35, 75))
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
    if rng.random() < probabilities.get("brightness_contrast", 0):
        array = np.asarray(result, dtype=np.float32)
        contrast = rng.uniform(0.75, 1.15)
        brightness = rng.uniform(-12, 12)
        result = Image.fromarray(np.clip(array * contrast + brightness, 0, 255).astype(np.uint8))
    return result
