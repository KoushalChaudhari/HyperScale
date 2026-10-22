from pathlib import Path

import torch
from PIL import Image

from evaluation.evaluate_srgan import evaluate_srgan
from inference.super_resolve import load_sr_generator, super_resolve
from training.pretrain import pretrain
from training.train_srgan import train_srgan


def _make_div2k_tree(root: Path) -> tuple[Path, Path]:
    hr_root = root / "train_hr"
    lr_root = root / "train_lr"
    hr_root.mkdir(parents=True)
    for scale in (2, 3, 4):
        (lr_root / f"X{scale}").mkdir(parents=True)
    for index in range(10):
        size = 36
        color = (30 + index * 12, 80 + index * 6, 130 - index * 5)
        Image.new("RGB", (size, size), color).save(hr_root / f"{index + 1:04d}.png")
        for scale in (2, 3, 4):
            Image.new("RGB", (size // scale, size // scale), color).save(
                lr_root / f"X{scale}" / f"{index + 1:04d}x{scale}.png"
            )
    return hr_root, lr_root


def test_two_stage_real_pair_training_evaluation_and_inference(tmp_path):
    hr_root, lr_root = _make_div2k_tree(tmp_path)
    config = {
        "seed": 7,
        "data": {
            "hr_dir": str(hr_root),
            "lr_root": str(lr_root),
            "scale": 4,
            "manifest": str(tmp_path / "split.json"),
            "pairing_report": str(tmp_path / "pairing.json"),
            "validation_fraction": 0.1,
            "lr_crop_size": 8,
        },
        "training": {
            "pretrain_epochs": 1,
            "srgan_epochs": 1,
            "batch_size": 2,
            "workers": 0,
            "generator_features": 4,
            "residual_blocks": 1,
            "generator_lr": 0.0001,
            "discriminator_lr": 0.0001,
            "adversarial_weight": 0.001,
            "content_weight": 1.0,
            "perceptual_weight": 0.0,
            "mixed_precision": False,
            "pretrain_limit_images": 2,
            "srgan_limit_images": 2,
            "max_minutes": 5,
            "checkpoint_interval": 1,
        },
        "paths": {
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        },
    }
    pretrain_result = pretrain(config, limit_images=2, lr_crop_size=8, features=4, residual_blocks=1, epochs=1)
    assert pretrain_result["global_step"] > 0
    resumed_pretrain = pretrain(
        config,
        limit_images=2,
        lr_crop_size=8,
        features=4,
        residual_blocks=1,
        epochs=2,
        resume=str(Path(config["paths"]["checkpoint_dir"]) / "generator_pretrain_latest.pth"),
    )
    assert resumed_pretrain["epochs_completed"] == 2
    result = train_srgan(
        config,
        limit_images=2,
        lr_crop_size=8,
        features=4,
        residual_blocks=1,
        epochs=1,
    )
    checkpoint = Path(config["paths"]["checkpoint_dir"]) / "srgan_generator_inference.pth"
    assert result["global_step"] > 0 and checkpoint.is_file()
    resumed_srgan = train_srgan(
        config,
        limit_images=2,
        lr_crop_size=8,
        features=4,
        residual_blocks=1,
        epochs=2,
        resume=str(Path(config["paths"]["checkpoint_dir"]) / "srgan_latest.pth"),
    )
    assert resumed_srgan["epochs_completed"] == 2
    model, device, metadata = load_sr_generator(checkpoint, "cpu", expected_scale=4)
    assert metadata["trained"] is True
    assert metadata["training_data_scope"] == "subset"
    source = Image.new("RGB", (9, 7), (100, 120, 140))
    restored = super_resolve(source, checkpoint, expected_scale=4, device=device)
    assert restored.image.size == (36, 28)
    assert restored.bicubic.size == restored.image.size
    report = evaluate_srgan(config, limit_images=1)
    assert report["pair_count"] == 1
    assert set(report["metrics"]) == {"bicubic", "pretrained_generator", "srgan"}
