"""Reference-based restoration metrics in the RGB [0, 255] domain."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Metric images must have matching shapes: {reference.shape} vs {candidate.shape}")
    if reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("Metrics require RGB images with shape [height, width, 3]")
    reference = np.asarray(reference, dtype=np.uint8)
    candidate = np.asarray(candidate, dtype=np.uint8)
    ssim = structural_similarity(reference, candidate, channel_axis=2, data_range=255)
    psnr = (
        float("inf")
        if np.array_equal(reference, candidate)
        else peak_signal_noise_ratio(reference, candidate, data_range=255)
    )
    return {"ssim": float(ssim), "psnr": float(psnr)}
