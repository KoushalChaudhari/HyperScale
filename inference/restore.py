"""Checkpoint-based inference independent from the Streamlit UI."""

from __future__ import annotations

import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image

from data.preprocessing import load_rgb_image, normalize_image, denormalize_image
from models import UNetGenerator


@dataclass
class RestorationResult:
    image: Image.Image
    elapsed_seconds: float
    device: str


def load_generator(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[UNetGenerator, torch.device, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained generator checkpoint not found: {path}")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        checkpoint = torch.load(path, map_location=selected_device, weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Unable to safely read generator checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("trained") is not True:
        raise ValueError("Checkpoint does not declare trained=True; train a model before restoration")
    state = checkpoint.get("generator")
    config = checkpoint.get("config", {})
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no generator weights: {path}")
    training = config.get("training", {}) if isinstance(config, dict) else {}
    base_channels = int(training.get("base_channels", checkpoint.get("base_channels", 32)))
    model = UNetGenerator(base_channels=base_channels)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Generator checkpoint is incompatible with RestoreAI U-Net: {exc}") from exc
    model.to(selected_device).eval()
    return model, selected_device, config


def _pad_tensor(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if not pad_h and not pad_w:
        return tensor, height, width
    padded = functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, height, width


def restore_image(
    image: Image.Image,
    checkpoint_path: str | Path | None = None,
    *,
    model: UNetGenerator | None = None,
    device: torch.device | None = None,
    max_side: int = 1024,
) -> RestorationResult:
    """Restore an RGB image in memory and resize back to its original dimensions."""
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Provide either a trained generator model or checkpoint_path")
        model, device, _ = load_generator(checkpoint_path, device)
    if device is None:
        device = next(model.parameters()).device
    original = load_rgb_image(image)
    original_size = original.size
    working = original
    scale = min(1.0, max_side / max(original.width, original.height))
    if scale < 1:
        working = original.resize(
            (max(1, round(original.width * scale)), max(1, round(original.height * scale))),
            Image.Resampling.LANCZOS,
        )
    tensor = normalize_image(working).unsqueeze(0).to(device)
    tensor, height, width = _pad_tensor(tensor, model.required_multiple)
    started = time.perf_counter()
    with torch.inference_mode():
        restored = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = denormalize_image(restored[0, :, :height, :width])
    if result.size != original_size:
        result = result.resize(original_size, Image.Resampling.LANCZOS)
    return RestorationResult(result.convert("RGB"), elapsed, str(device))
