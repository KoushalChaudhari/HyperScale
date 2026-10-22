"""Scale-checked X2/X4 SRGAN inference."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image

from data.preprocessing import denormalize_image, load_rgb_image, normalize_image
from models.sr_generator import SRGenerator


@dataclass
class SuperResolutionResult:
    image: Image.Image
    bicubic: Image.Image
    elapsed_seconds: float
    device: str
    scale: int


def load_sr_generator(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    expected_scale: int | None = None,
) -> tuple[SRGenerator, torch.device, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained SRGAN checkpoint not found: {path}")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        payload = torch.load(path, map_location=selected_device, weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not safely load SRGAN checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("trained") is not True:
        raise ValueError("Checkpoint has no confirmed trained SRGAN weights")
    if payload.get("model_type") != "srgan":
        raise ValueError("Checkpoint is not an SRGAN generator; old restoration/Pix2Pix weights are incompatible")
    required = {"scale", "features", "residual_blocks", "generator"}
    if not required.issubset(payload):
        raise ValueError(f"SRGAN checkpoint is missing required fields: {sorted(required - payload.keys())}")
    scale = int(payload.get("scale", -1))
    if scale not in (2, 4):
        raise ValueError(f"Checkpoint uses unsupported scale X{scale}; supported scales are X2 and X4")
    if expected_scale is not None and scale != expected_scale:
        raise ValueError(f"Checkpoint scale X{scale} does not match requested scale X{expected_scale}")
    generator = SRGenerator(
        scale=scale,
        features=int(payload["features"]),
        residual_blocks=int(payload["residual_blocks"]),
    )
    try:
        generator.load_state_dict(payload["generator"], strict=True)
    except (KeyError, RuntimeError) as exc:
        raise ValueError(f"Invalid SRGAN generator weights in {path}: {exc}") from exc
    generator.to(selected_device).eval()
    return generator, selected_device, payload


def bicubic_upscale(image: Image.Image, scale: int) -> Image.Image:
    if scale not in (2, 4):
        raise ValueError(f"Bicubic baseline supports X2/X4 only, not X{scale}")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.BICUBIC)


def super_resolve(
    image: Image.Image,
    checkpoint_path: str | Path,
    *,
    expected_scale: int | None = None,
    device: str | torch.device | None = None,
    max_input_side: int = 768,
) -> SuperResolutionResult:
    """Run trained SRGAN inference; reject overly large inputs instead of changing scale."""
    generator, selected_device, _ = load_sr_generator(checkpoint_path, device, expected_scale)
    lr = load_rgb_image(image)
    if max(lr.size) > max_input_side:
        raise ValueError(
            f"Input is {lr.width}x{lr.height}; the configured maximum LR side is {max_input_side}px. "
            "Downscale the LR image before upload or raise max_input_side if memory allows; "
            "tiled SR inference is not implemented."
        )
    tensor = normalize_image(lr).unsqueeze(0).to(selected_device)
    started = time.perf_counter()
    with torch.inference_mode():
        sr = generator(tensor)
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    elapsed = time.perf_counter() - started
    result = denormalize_image(sr[0].cpu()).convert("RGB")
    expected_size = (lr.width * generator.scale, lr.height * generator.scale)
    if result.size != expected_size:
        raise RuntimeError(f"SR output dimensions {result.size} do not match expected {expected_size}")
    return SuperResolutionResult(
        image=result,
        bicubic=bicubic_upscale(lr, generator.scale),
        elapsed_seconds=elapsed,
        device=str(selected_device),
        scale=generator.scale,
    )
