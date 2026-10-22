"""Pixel-loss pretraining of the X2/X4 SRGAN generator."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from training.srgan_utils import (
    ROOT,
    append_sr_log,
    build_generator,
    checkpoint_path,
    elapsed,
    ensure_manifest,
    inference_payload,
    load_config,
    load_stage_checkpoint,
    make_loader,
    resolve_path,
    save_preview,
    save_stage_checkpoint,
    seed_everything,
    set_dataset_epoch,
    validate_generator,
)

LOGGER = logging.getLogger("restoreai.srgan.pretrain")


def pretrain(
    config: dict,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit_images: int | None = None,
    lr_crop_size: int | None = None,
    features: int | None = None,
    residual_blocks: int | None = None,
    max_minutes: float | None = None,
    resume: str | None = None,
) -> dict:
    training = config["training"]
    data = config["data"]
    if epochs is not None:
        training["pretrain_epochs"] = epochs
    if batch_size is not None:
        training["batch_size"] = batch_size
    if lr_crop_size is not None:
        data["lr_crop_size"] = lr_crop_size
    if features is not None:
        training["generator_features"] = features
    if residual_blocks is not None:
        training["residual_blocks"] = residual_blocks
    if max_minutes is not None:
        training["max_minutes"] = max_minutes
    budget = float(training.get("max_minutes", 60))
    if not 0 < budget <= 60:
        raise ValueError("max_minutes must be greater than 0 and no more than 60")

    seed_everything(int(config["seed"]))
    ensure_manifest(config)
    selected_limit = (
        limit_images if limit_images is not None else training.get("pretrain_limit_images")
    )
    train_loader = make_loader(config, "train", training=True, limit_images=selected_limit)
    val_loader = make_loader(config, "validation", training=False, limit_images=selected_limit)
    training["run_pair_limit"] = selected_limit
    training["run_pair_count"] = len(train_loader.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    generator = build_generator(config).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=float(training["generator_lr"]), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    start_epoch, global_step = 0, 0
    best_psnr = float("-inf")
    if resume:
        checkpoint = load_stage_checkpoint(
            resume,
            stage="pretrain",
            generator=generator,
            discriminator=None,
            optimizer_g=optimizer,
            optimizer_d=None,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_psnr = float(checkpoint["best_validation_psnr"])

    output_dir = resolve_path(config, "output_dir")
    log_path = resolve_path(config, "log_dir") / "srgan_training.csv"
    checkpoint_path(config, "generator_pretrain_latest").parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best_val = None
    epoch_completed = start_epoch
    stopped_for_budget = False
    for epoch in range(start_epoch, int(training["pretrain_epochs"])):
        generator.train()
        set_dataset_epoch(train_loader, epoch)
        total_loss, steps = 0.0, 0
        preview = None
        progress = tqdm(train_loader, desc=f"SRGAN pretrain {epoch + 1}", leave=False)
        for batch in progress:
            if elapsed(started) >= budget * 60:
                stopped_for_budget = True
                break
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            preview = batch
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                sr = generator(lr)
                if sr.shape != hr.shape:
                    raise RuntimeError(f"Generator output {tuple(sr.shape)} does not match HR target {tuple(hr.shape)}")
                loss = torch.nn.functional.l1_loss(sr, hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            steps += 1
            global_step += 1
            progress.set_postfix(l1=f"{float(loss.detach()):.4f}")
        if steps == 0 and not stopped_for_budget:
            raise ValueError("Training loader produced no batches")
        if global_step == 0:
            raise RuntimeError("Training budget expired before any optimizer update; no trained checkpoint was saved")

        validation = validate_generator(generator, val_loader, device)
        epoch_completed = epoch + 1
        best_val = validation
        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            save_stage_checkpoint(
                checkpoint_path(config, "generator_pretrain_best"),
                stage="pretrain",
                generator=generator,
                discriminator=None,
                optimizer_g=optimizer,
                optimizer_d=None,
                epoch=epoch_completed,
                global_step=global_step,
                best_validation_psnr=best_psnr,
                config=config,
                elapsed_seconds=elapsed(started),
            )
            torch.save(
                inference_payload(generator, config, epoch_completed, "pretrained"),
                checkpoint_path(config, "generator_pretrain_inference"),
            )
        save_stage_checkpoint(
            checkpoint_path(config, "generator_pretrain_latest"),
            stage="pretrain",
            generator=generator,
            discriminator=None,
            optimizer_g=optimizer,
            optimizer_d=None,
            epoch=epoch_completed,
            global_step=global_step,
            best_validation_psnr=best_psnr,
            config=config,
            elapsed_seconds=elapsed(started),
        )
        append_sr_log(
            log_path,
            {
                "stage": "pretrain",
                "epoch": epoch_completed,
                "global_step": global_step,
                "generator_loss": total_loss / max(1, steps),
                "reconstruction_loss": total_loss / max(1, steps),
                "validation_psnr": validation["psnr"],
                "validation_ssim": validation["ssim"],
                "elapsed_seconds": elapsed(started),
            },
        )
        if preview is not None:
            save_preview(generator, preview, device, output_dir / "srgan" / f"pretrain_epoch_{epoch_completed:04d}.png")
        LOGGER.info(
            "pretrain epoch=%d/%d L1=%.5f val_PSNR=%.3f val_SSIM=%.5f",
            epoch_completed,
            training["pretrain_epochs"],
            total_loss / max(1, steps),
            validation["psnr"],
            validation["ssim"],
        )
        if stopped_for_budget:
            break

    return {
        "stage": "pretrain",
        "device": str(device),
        "scale": int(data["scale"]),
        "epochs_completed": epoch_completed,
        "global_step": global_step,
        "best_validation_psnr": best_psnr if math.isfinite(best_psnr) else None,
        "last_validation": best_val,
        "elapsed_seconds": elapsed(started),
        "stopped_for_budget": stopped_for_budget,
        "checkpoint": str(checkpoint_path(config, "generator_pretrain_best")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--lr-crop-size", type=int)
    parser.add_argument("--features", type=int)
    parser.add_argument("--residual-blocks", type=int)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--resume")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = pretrain(
            load_config(args.config),
            epochs=args.epochs,
            batch_size=args.batch_size,
            limit_images=args.limit_images,
            lr_crop_size=args.lr_crop_size,
            features=args.features,
            residual_blocks=args.residual_blocks,
            max_minutes=args.max_minutes,
            resume=args.resume,
        )
        LOGGER.info("Pretraining summary: %s", summary)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("Out of memory; lower batch size, crop size, or generator features") from exc
        raise


if __name__ == "__main__":
    main()
