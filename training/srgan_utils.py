"""Shared SRGAN configuration, data, validation, and checkpoint helpers."""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

from data.div2k_dataset import DIV2KPairedDataset, build_manifest
from data.preprocessing import denormalize_image
from evaluation.metrics import image_metrics
from models.sr_generator import SRGenerator

LOGGER = logging.getLogger("restoreai.srgan")
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("SRGAN config must be a YAML mapping with a data section")
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else ROOT / path


def checkpoint_path(config: dict[str, Any], stem: str) -> Path:
    scale = int(config["data"]["scale"])
    scale_suffix = "" if scale == 4 else f"_X{scale}"
    return resolve_path(config, "checkpoint_dir") / f"{stem}{scale_suffix}.pth"


def resolve_data_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["data"][key])
    return path if path.is_absolute() else ROOT / path


def ensure_manifest(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    manifest = resolve_data_path(config, "manifest")
    report = resolve_data_path(config, "pairing_report")
    if manifest.exists() and not force:
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        expected = {
            "seed": int(config["seed"]),
            "validation_fraction": float(config["data"]["validation_fraction"]),
            "selected_scale": int(config["data"]["scale"]),
            "hr_root": str(resolve_data_path(config, "hr_dir").resolve()),
            "lr_root": str(resolve_data_path(config, "lr_root").resolve()),
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Saved DIV2K manifest does not match current data/split configuration: {mismatches}. "
                "Rebuild intentionally with `python -m data.build_manifest --force`."
            )
        return existing
    result = build_manifest(
        resolve_data_path(config, "hr_dir"),
        resolve_data_path(config, "lr_root"),
        manifest,
        report,
        scale=int(config["data"]["scale"]),
        validation_fraction=float(config["data"]["validation_fraction"]),
        seed=int(config["seed"]),
        force=force,
    )
    return result["manifest"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(
    config: dict[str, Any],
    split: str,
    *,
    training: bool,
    limit_images: int | None = None,
) -> DataLoader:
    manifest = resolve_data_path(config, "manifest")
    scale = int(config["data"]["scale"])
    dataset = DIV2KPairedDataset(
        manifest,
        scale=scale,
        split=split,
        lr_crop_size=int(config["data"]["lr_crop_size"]),
        seed=int(config["seed"]),
    )
    if limit_images is not None:
        if limit_images < 1:
            raise ValueError("--limit-images must be positive")
        dataset = Subset(dataset, range(min(limit_images, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=training,
        num_workers=int(config["training"]["workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def set_dataset_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    if isinstance(dataset, DIV2KPairedDataset):
        dataset.set_epoch(epoch)


def build_generator(config: dict[str, Any]) -> SRGenerator:
    training = config["training"]
    return SRGenerator(
        scale=int(config["data"]["scale"]),
        features=int(training["generator_features"]),
        residual_blocks=int(training["residual_blocks"]),
    )


@torch.inference_mode()
def validate_generator(
    generator: SRGenerator,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    totals = {"psnr": 0.0, "ssim": 0.0}
    count = 0
    for batch in loader:
        lr, hr = batch["lr"].to(device), batch["hr"]
        sr = generator(lr).cpu()
        if sr.shape != hr.shape:
            raise RuntimeError(f"SR output {tuple(sr.shape)} does not match HR target {tuple(hr.shape)}")
        for index in range(sr.shape[0]):
            result = image_metrics(np.asarray(denormalize_image(hr[index])), np.asarray(denormalize_image(sr[index])))
            totals["psnr"] += result["psnr"]
            totals["ssim"] += result["ssim"]
            count += 1
    if not count:
        raise ValueError("Validation split has no pairs")
    generator.train()
    return {name: value / count for name, value in totals.items()}


def inference_payload(generator: SRGenerator, config: dict[str, Any], epoch: int, stage: str) -> dict[str, Any]:
    training = config.get("training", {})
    return {
        "format_version": 1,
        "model_type": "srgan",
        "trained": True,
        "stage": stage,
        "scale": generator.scale,
        "features": generator.features,
        "residual_blocks": generator.residual_blocks,
        "generator": generator.state_dict(),
        "epoch": epoch,
        "training_data_scope": "subset" if training.get("run_pair_limit") else "full_manifest",
        "training_pair_limit": training.get("run_pair_limit"),
        "training_pair_count": training.get("run_pair_count"),
        "config": config,
    }


def append_sr_log(path: Path, row: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "epoch",
        "global_step",
        "generator_loss",
        "discriminator_loss",
        "reconstruction_loss",
        "adversarial_loss",
        "perceptual_loss",
        "validation_psnr",
        "validation_ssim",
        "elapsed_seconds",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def save_stage_checkpoint(
    path: Path,
    *,
    stage: str,
    generator: SRGenerator,
    discriminator: torch.nn.Module | None,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    best_validation_psnr: float,
    config: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "stage": stage,
            "scale": generator.scale,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict() if discriminator is not None else None,
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_psnr": best_validation_psnr,
            "config": config,
            "elapsed_seconds": elapsed_seconds,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_stage_checkpoint(
    path: str | Path,
    *,
    stage: str,
    generator: SRGenerator,
    discriminator: torch.nn.Module | None,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("stage") != stage:
        raise ValueError(f"Checkpoint is not a {stage} SRGAN checkpoint: {path}")
    if int(checkpoint.get("scale", -1)) != generator.scale:
        raise ValueError("Checkpoint scale does not match the configured scale")
    generator.load_state_dict(checkpoint["generator"], strict=True)
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if discriminator is not None:
        if checkpoint.get("discriminator") is None or optimizer_d is None:
            raise ValueError("Adversarial resume checkpoint is missing discriminator state")
        discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint


def save_preview(generator: SRGenerator, batch: dict[str, Any], device: torch.device, path: Path) -> None:
    generator.eval()
    with torch.inference_mode():
        output = generator(batch["lr"][:4].to(device)).cpu()
    strips = []
    for index in range(output.shape[0]):
        lr = denormalize_image(batch["lr"][index])
        bicubic = lr.resize((lr.width * generator.scale, lr.height * generator.scale), Image.Resampling.BICUBIC)
        sr = denormalize_image(output[index])
        hr = denormalize_image(batch["hr"][index])
        strip = Image.new("RGB", (hr.width * 3, hr.height))
        for column, image in enumerate((bicubic, sr, hr)):
            strip.paste(image, (column * hr.width, 0))
        strips.append(strip)
    canvas = Image.new("RGB", (strips[0].width, sum(image.height for image in strips)))
    top = 0
    for strip in strips:
        canvas.paste(strip, (0, top))
        top += strip.height
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    generator.train()


def elapsed(started: float) -> float:
    return time.monotonic() - started
