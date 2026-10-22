"""Checkpoint serialization and restoration helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    best_validation_ssim: float,
    scaler_g: Any = None,
    scaler_d: Any = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "trained": global_step > 0,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scaler_g": scaler_g.state_dict() if scaler_g is not None else None,
        "scaler_d": scaler_d.state_dict() if scaler_d is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "best_validation_ssim": best_validation_ssim,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, destination)


def load_training_checkpoint(
    path: str | Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    scaler_g: Any = None,
    scaler_d: Any = None,
) -> dict[str, Any]:
    """Load a trusted project training checkpoint and restore optimizer/RNG state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"generator", "discriminator", "optimizer_g", "optimizer_d", "epoch", "global_step"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Not a RestoreAI training checkpoint: {path}")
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    if scaler_g is not None and checkpoint.get("scaler_g"):
        scaler_g.load_state_dict(checkpoint["scaler_g"])
    if scaler_d is not None and checkpoint.get("scaler_d"):
        scaler_d.load_state_dict(checkpoint["scaler_d"])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint
