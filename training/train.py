"""Reproducible time-budgeted Pix2Pix training entry point."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import CleanImageDataset, PairedImageDataset, create_source_split, split_clean_dataset
from data.preprocessing import denormalize_image
from evaluation.metrics import image_metrics
from models import PatchDiscriminator, UNetGenerator
from training.checkpoint import load_training_checkpoint, save_checkpoint
from training.logger import CSVLogger
from training.losses import discriminator_loss, generator_losses

LOGGER = logging.getLogger("restoreai.train")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    config: dict[str, Any],
    split: str,
    training: bool,
) -> DataLoader:
    data = config["data"]
    manifest_path = data["split_manifest"]
    size = int(data["image_size"])
    if bool(data.get("paired_input_dir")) != bool(data.get("paired_target_dir")):
        raise ValueError("Configure both paired_input_dir and paired_target_dir, or neither")
    if data.get("paired_input_dir") and data.get("paired_target_dir"):
        pairs = PairedImageDataset(data["paired_input_dir"], data["paired_target_dir"], size)
        input_paths = [pair[0] for pair in pairs.pairs]
        source_split = create_source_split(
            input_paths,
            manifest_path,
            seed=int(config["seed"]),
            fractions=(
                float(data["train_fraction"]),
                float(data["validation_fraction"]),
                float(data["test_fraction"]),
            ),
        )
        positions = {str(path.resolve()): index for index, path in enumerate(input_paths)}
        indices = [positions[path] for path in source_split[split]]
        dataset = Subset(pairs, indices)
    else:
        source_split = split_clean_dataset(
            data["clean_dir"],
            manifest_path,
            int(config["seed"]),
            (
                float(data["train_fraction"]),
                float(data["validation_fraction"]),
                float(data["test_fraction"]),
            ),
        )
        limit = data.get(f"limit_{split}")
        selected = source_split[split][: int(limit)] if limit else source_split[split]
        dataset = CleanImageDataset(
            selected,
            size,
            int(config["seed"]) + (split == "test") * 100_000,
            training,
            data.get("degradation_probabilities"),
        )
    if len(dataset) == 0:
        raise ValueError(f"No images in the {split} split")
    batch_size = int(config["training"]["batch_size"]) if training else 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=int(config["training"]["workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=training and len(dataset) >= batch_size,
    )


@torch.inference_mode()
def _validate(generator: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    generator.eval()
    totals = {"ssim": 0.0, "psnr": 0.0}
    count = 0
    for batch in loader:
        degraded = batch["input"].to(device)
        target = batch["target"]
        restored = generator(degraded).cpu()
        for index in range(restored.shape[0]):
            target_image = np.asarray(denormalize_image(target[index]))
            restored_image = np.asarray(denormalize_image(restored[index]))
            metrics = image_metrics(target_image, restored_image)
            totals["ssim"] += metrics["ssim"]
            totals["psnr"] += metrics["psnr"]
            count += 1
    if not count:
        raise ValueError("Validation split is empty")
    generator.train()
    return {key: value / count for key, value in totals.items()}


def _save_preview(generator: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device, path: Path) -> None:
    generator.eval()
    with torch.inference_mode():
        restored = generator(batch["input"][:4].to(device)).cpu()
    samples = []
    for index in range(restored.shape[0]):
        row = [denormalize_image(batch[key][index]) for key in ("input", "target")]
        row.append(denormalize_image(restored[index]))
        canvas = Image.new("RGB", (row[0].width * 3, row[0].height))
        for column, image in enumerate(row):
            canvas.paste(image, (column * image.width, 0))
        samples.append(canvas)
    sheet = Image.new("RGB", (samples[0].width, sum(item.height for item in samples)))
    offset = 0
    for sample in samples:
        sheet.paste(sample, (0, offset))
        offset += sample.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    generator.train()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    return config


def train(config: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    _seed_everything(int(config["seed"]))
    training = config["training"]
    if int(config["data"]["image_size"]) % 128:
        raise ValueError("data.image_size must be a multiple of 128 for the configured U-Net depth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_minutes = float(training.get("max_minutes", 60))
    if not 0 < configured_minutes <= 60:
        raise ValueError("training.max_minutes must be greater than 0 and no more than 60")
    amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    LOGGER.info("Using %s; mixed precision=%s", device, amp_enabled)

    train_loader = _loader(config, "train", True)
    validation_loader = _loader(config, "validation", False)
    generator = UNetGenerator(base_channels=int(training["base_channels"])).to(device)
    discriminator = PatchDiscriminator(base_channels=int(training["base_channels"])).to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scaler_d = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_ssim = float("-inf")
    if resume:
        state = load_training_checkpoint(
            resume, generator, discriminator, optimizer_g, optimizer_d, device, scaler_g, scaler_d
        )
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_ssim = float(state.get("best_validation_ssim", float("-inf")))
        LOGGER.info("Resumed at epoch %d, step %d", start_epoch, global_step)

    checkpoint_dir = Path(config["checkpoint_dir"])
    output_dir = Path(config["output_dir"])
    log_dir = Path(config["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(log_dir / "training.csv")
    max_seconds = configured_minutes * 60
    started = time.monotonic()
    last_validation: dict[str, float] | None = None
    stopped_for_budget = False
    epochs_without_improvement = 0
    completed_epoch = start_epoch
    epochs = int(training["epochs"])

    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        sums = {"generator": 0.0, "discriminator": 0.0, "adversarial": 0.0, "reconstruction": 0.0}
        batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        preview_batch: dict[str, torch.Tensor] | None = None
        for batch in progress:
            if time.monotonic() - started >= max_seconds:
                stopped_for_budget = True
                break
            degraded = batch["input"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            preview_batch = batch

            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                with torch.no_grad():
                    generated_for_d = generator(degraded)
                real_logits = discriminator(degraded, target)
                fake_logits = discriminator(degraded, generated_for_d)
                loss_d = discriminator_loss(criterion, real_logits, fake_logits)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                generated = generator(degraded)
                fake_logits = discriminator(degraded, generated)
                loss_g, loss_adv, loss_l1 = generator_losses(
                    criterion, fake_logits, generated, target, float(training["lambda_l1"])
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            global_step += 1
            batches += 1
            sums["generator"] += float(loss_g.detach())
            sums["discriminator"] += float(loss_d.detach())
            sums["adversarial"] += float(loss_adv.detach())
            sums["reconstruction"] += float(loss_l1.detach())
            progress.set_postfix(g=f"{float(loss_g.detach()):.3f}", d=f"{float(loss_d.detach()):.3f}")

        elapsed = time.monotonic() - started
        if batches == 0 and not stopped_for_budget:
            raise ValueError("Training loader produced no batches")
        validation = _validate(generator, validation_loader, device)
        last_validation = validation
        epoch_number = epoch + 1
        completed_epoch = epoch_number
        if validation["ssim"] > best_ssim:
            best_ssim = validation["ssim"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "config": config,
                    "epoch": epoch_number,
                    "validation_ssim": best_ssim,
                    "trained": global_step > 0,
                },
                checkpoint_dir / "best_generator.pth",
            )
        else:
            epochs_without_improvement += 1
        if epoch_number % int(training["checkpoint_interval"]) == 0 or stopped_for_budget:
            save_checkpoint(
                checkpoint_dir / "latest.pth",
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                epoch_number,
                global_step,
                config,
                best_ssim,
                scaler_g,
                scaler_d,
            )
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "config": config,
                    "epoch": epoch_number,
                    "trained": global_step > 0,
                },
                checkpoint_dir / "generator_latest.pth",
            )
        denominator = max(1, batches)
        logger.log(
            {
                "epoch": epoch_number,
                "global_step": global_step,
                "generator_loss": sums["generator"] / denominator,
                "discriminator_loss": sums["discriminator"] / denominator,
                "adversarial_loss": sums["adversarial"] / denominator,
                "reconstruction_loss": sums["reconstruction"] / denominator,
                "validation_ssim": validation["ssim"],
                "validation_psnr": validation["psnr"],
                "elapsed_seconds": elapsed,
            }
        )
        if preview_batch is not None and epoch_number % int(training["sample_interval"]) == 0:
            _save_preview(generator, preview_batch, device, output_dir / f"epoch_{epoch_number:04d}.png")
        LOGGER.info(
            "epoch=%d g=%.4f d=%.4f val_ssim=%.4f val_psnr=%.2f elapsed=%.1fs",
            epoch_number,
            sums["generator"] / denominator,
            sums["discriminator"] / denominator,
            validation["ssim"],
            validation["psnr"],
            elapsed,
        )
        if stopped_for_budget:
            LOGGER.warning("Reached configured training time budget; latest state was saved")
            break
        patience = int(training.get("patience", 0))
        if patience > 0 and epochs_without_improvement >= patience:
            LOGGER.info("Early stopping after %d epochs without SSIM improvement", patience)
            break

    elapsed = time.monotonic() - started
    if last_validation and not math.isfinite(last_validation["psnr"]):
        last_validation = {**last_validation, "psnr": "Infinity"}
    summary = {
        "device": str(device),
        "epochs_completed": completed_epoch,
        "global_step": global_step,
        "training_elapsed_seconds": elapsed,
        "time_budget_minutes": float(training.get("max_minutes", 60)),
        "best_validation_ssim": best_ssim if best_ssim != float("-inf") else None,
        "last_validation": last_validation,
        "stopped_for_budget": stopped_for_budget,
        "test_evaluated": False,
    }
    (log_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_checkpoint(
        checkpoint_dir / "latest.pth",
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        int(summary["epochs_completed"]),
        global_step,
        config,
        best_ssim,
        scaler_g,
        scaler_d,
    )
    torch.save(
        {
            "generator": generator.state_dict(),
            "config": config,
            "epoch": summary["epochs_completed"],
            "trained": global_step > 0,
        },
        checkpoint_dir / "generator_latest.pth",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--clean-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--resume")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = _load_config(Path(args.config))
    if config.get("project") == "srgan":
        raise SystemExit(
            "configs/default.yaml now targets SRGAN. Use `python -m training.pretrain` followed by "
            "`python -m training.train_srgan`."
        )
    if args.clean_dir:
        config["data"]["clean_dir"] = args.clean_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.output_dir:
        config["output_dir"] = args.output_dir
        config["checkpoint_dir"] = str(Path(args.output_dir) / "checkpoints")
        config["log_dir"] = str(Path(args.output_dir) / "logs")
    if args.max_minutes is not None:
        config["training"]["max_minutes"] = args.max_minutes
    try:
        summary = train(config, args.resume)
        LOGGER.info("Training finished: %s", json.dumps(summary))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("Out of memory; reduce batch_size, image_size, or base_channels") from exc
        raise


if __name__ == "__main__":
    main()
