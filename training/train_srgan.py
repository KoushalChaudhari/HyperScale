"""Adversarial fine-tuning stage for the paired-image SRGAN."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from models.sr_discriminator import SRDiscriminator
from training.losses import VGG19ContentLoss
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

LOGGER = logging.getLogger("restoreai.srgan.adversarial")


def train_srgan(
    config: dict,
    *,
    pretrained: str | None = None,
    resume: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    limit_images: int | None = None,
    lr_crop_size: int | None = None,
    features: int | None = None,
    residual_blocks: int | None = None,
    max_minutes: float | None = None,
) -> dict:
    training, data = config["training"], config["data"]
    if epochs is not None:
        training["srgan_epochs"] = epochs
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

    perceptual_weight = float(training.get("perceptual_weight", 0))
    content_loss = None
    if perceptual_weight > 0:
        vgg_path = training.get("vgg19_weights_path")
        if not vgg_path or not Path(vgg_path).is_file():
            raise FileNotFoundError(
                "Perceptual loss was enabled but a local ImageNet VGG19 weights file was not found. "
                "Set training.vgg19_weights_path to a verified full VGG19 state dict, or set perceptual_weight: 0."
            )
        content_loss = VGG19ContentLoss(vgg_path)

    seed_everything(int(config["seed"]))
    ensure_manifest(config)
    selected_limit = (
        limit_images if limit_images is not None else training.get("srgan_limit_images")
    )
    train_loader = make_loader(config, "train", training=True, limit_images=selected_limit)
    val_loader = make_loader(config, "validation", training=False, limit_images=selected_limit)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = bool(training.get("mixed_precision", True)) and device.type == "cuda"

    generator = build_generator(config).to(device)
    discriminator = SRDiscriminator(int(training["generator_features"])).to(device)
    if content_loss is not None:
        content_loss = content_loss.to(device).eval()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=float(training["generator_lr"]), betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=float(training["discriminator_lr"]), betas=(0.9, 0.999)
    )
    scaler_g = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    scaler_d = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    start_epoch, global_step = 0, 0
    best_psnr = float("-inf")
    if resume:
        checkpoint = load_stage_checkpoint(
            resume,
            stage="srgan",
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_psnr = float(checkpoint["best_validation_psnr"])
    else:
        pretrained_path = (
            Path(pretrained) if pretrained else checkpoint_path(config, "generator_pretrain_best")
        )
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Pretrained generator checkpoint not found: {pretrained_path}. Run python -m training.pretrain first."
            )
        pretrained_state = torch.load(pretrained_path, map_location=device, weights_only=False)
        if not isinstance(pretrained_state, dict) or pretrained_state.get("stage") != "pretrain":
            raise ValueError("The supplied checkpoint is not a RestoreAI generator-pretraining checkpoint")
        if int(pretrained_state.get("scale", -1)) != generator.scale:
            raise ValueError("Pretraining checkpoint scale differs from configured SRGAN scale")
        try:
            generator.load_state_dict(pretrained_state["generator"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise ValueError("Pretraining checkpoint architecture does not match the configured SRGAN model") from exc
        training["pretraining_epochs_completed"] = pretrained_state.get("epoch")
        training["pretraining_pair_limit"] = pretrained_state.get("config", {}).get("training", {}).get(
            "run_pair_limit"
        )
        training["pretraining_pair_count"] = pretrained_state.get("config", {}).get("training", {}).get(
            "run_pair_count"
        )

    checkpoint_dir = resolve_path(config, "checkpoint_dir")
    output_dir = resolve_path(config, "output_dir") / "srgan"
    log_path = resolve_path(config, "log_dir") / "srgan_training.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best_validation = None
    stopped_for_budget = False
    epoch_completed = start_epoch
    criterion = torch.nn.BCEWithLogitsLoss()
    training["run_pair_limit"] = selected_limit
    training["run_pair_count"] = len(train_loader.dataset)
    for epoch in range(start_epoch, int(training["srgan_epochs"])):
        generator.train()
        discriminator.train()
        if content_loss is not None:
            content_loss.eval()
        set_dataset_epoch(train_loader, epoch)
        totals = {"g": 0.0, "d": 0.0, "rec": 0.0, "adv": 0.0, "perceptual": 0.0}
        steps = 0
        preview = None
        for batch in tqdm(train_loader, desc=f"SRGAN adversarial {epoch + 1}", leave=False):
            if elapsed(started) >= budget * 60:
                stopped_for_budget = True
                break
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            preview = batch

            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                with torch.no_grad():
                    fake_for_d = generator(lr)
                real_logits = discriminator(hr)
                fake_logits = discriminator(fake_for_d.detach())
                loss_d = 0.5 * (
                    criterion(real_logits, torch.ones_like(real_logits))
                    + criterion(fake_logits, torch.zeros_like(fake_logits))
                )
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                generated = generator(lr)
                logits = discriminator(generated)
                loss_adv = criterion(logits, torch.ones_like(logits))
                loss_rec = torch.nn.functional.l1_loss(generated, hr)
                loss_perceptual = (
                    content_loss(generated, hr) if content_loss is not None else generated.new_zeros(())
                )
                loss_g = (
                    float(training["content_weight"]) * loss_rec
                    + float(training["adversarial_weight"]) * loss_adv
                    + perceptual_weight * loss_perceptual
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            steps += 1
            global_step += 1
            totals["g"] += float(loss_g.detach())
            totals["d"] += float(loss_d.detach())
            totals["rec"] += float(loss_rec.detach())
            totals["adv"] += float(loss_adv.detach())
            totals["perceptual"] += float(loss_perceptual.detach())

        if steps == 0 and not stopped_for_budget:
            raise ValueError("Training loader produced no batches")
        if global_step == 0:
            raise RuntimeError("Training budget expired before any optimizer update; no trained checkpoint was saved")
        validation = validate_generator(generator, val_loader, device)
        best_validation = validation
        epoch_completed = epoch + 1
        checkpoint_args = {
            "stage": "srgan",
            "generator": generator,
            "discriminator": discriminator,
            "optimizer_g": optimizer_g,
            "optimizer_d": optimizer_d,
            "epoch": epoch_completed,
            "global_step": global_step,
            "config": config,
            "elapsed_seconds": elapsed(started),
        }
        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            save_stage_checkpoint(
                checkpoint_path(config, "srgan_best"),
                best_validation_psnr=best_psnr,
                **checkpoint_args,
            )
            torch.save(
                inference_payload(generator, config, epoch_completed, "srgan"),
                checkpoint_path(config, "srgan_generator_inference"),
            )
        save_stage_checkpoint(
            checkpoint_path(config, "srgan_latest"),
            best_validation_psnr=best_psnr,
            **checkpoint_args,
        )
        append_sr_log(
            log_path,
            {
                "stage": "srgan",
                "epoch": epoch_completed,
                "global_step": global_step,
                "generator_loss": totals["g"] / max(1, steps),
                "discriminator_loss": totals["d"] / max(1, steps),
                "reconstruction_loss": totals["rec"] / max(1, steps),
                "adversarial_loss": totals["adv"] / max(1, steps),
                "perceptual_loss": totals["perceptual"] / max(1, steps),
                "validation_psnr": validation["psnr"],
                "validation_ssim": validation["ssim"],
                "elapsed_seconds": elapsed(started),
            },
        )
        if preview is not None:
            save_preview(generator, preview, device, output_dir / f"srgan_epoch_{epoch_completed:04d}.png")
        LOGGER.info(
            "srgan epoch=%d/%d G=%.5f D=%.5f val_PSNR=%.3f val_SSIM=%.5f",
            epoch_completed,
            training["srgan_epochs"],
            totals["g"] / max(1, steps),
            totals["d"] / max(1, steps),
            validation["psnr"],
            validation["ssim"],
        )
        if stopped_for_budget:
            break

    return {
        "stage": "srgan",
        "device": str(device),
        "scale": generator.scale,
        "epochs_completed": epoch_completed,
        "global_step": global_step,
        "best_validation_psnr": best_psnr if math.isfinite(best_psnr) else None,
        "last_validation": best_validation,
        "elapsed_seconds": elapsed(started),
        "stopped_for_budget": stopped_for_budget,
        "checkpoint": str(checkpoint_path(config, "srgan_generator_inference")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--lr-crop-size", type=int)
    parser.add_argument("--features", type=int)
    parser.add_argument("--residual-blocks", type=int)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = train_srgan(
            load_config(args.config),
            pretrained=args.pretrained,
            resume=args.resume,
            epochs=args.epochs,
            batch_size=args.batch_size,
            limit_images=args.limit_images,
            lr_crop_size=args.lr_crop_size,
            features=args.features,
            residual_blocks=args.residual_blocks,
            max_minutes=args.max_minutes,
        )
        LOGGER.info("SRGAN summary: %s", summary)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError("Out of memory; lower batch size, crop size, or generator features") from exc
        raise


if __name__ == "__main__":
    main()
