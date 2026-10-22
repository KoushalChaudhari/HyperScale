from pathlib import Path

import torch
from PIL import Image

from evaluation.evaluate import evaluate
from models import PatchDiscriminator, UNetGenerator
from training.checkpoint import load_training_checkpoint
from training.train import train


def test_one_batch_cpu_training_smoke(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    for index in range(10):
        image = Image.new("RGB", (128, 128), (40 + index * 10, 90, 150))
        image.save(clean / f"image_{index:02d}.png")
    config = {
        "seed": 11,
        "data": {
            "clean_dir": str(clean),
            "paired_input_dir": None,
            "paired_target_dir": None,
            "split_manifest": str(tmp_path / "source_split.json"),
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
            "image_size": 128,
            "limit_train": None,
            "limit_validation": None,
            "limit_test": None,
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "workers": 0,
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "base_channels": 2,
            "lambda_l1": 10.0,
            "checkpoint_interval": 1,
            "sample_interval": 1,
            "max_minutes": 60,
            "mixed_precision": False,
            "patience": 0,
        },
        "output_dir": str(tmp_path / "outputs"),
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "log_dir": str(tmp_path / "logs"),
    }
    summary = train(config)
    assert summary["global_step"] >= 1
    assert summary["test_evaluated"] is False
    assert Path(config["checkpoint_dir"], "best_generator.pth").is_file()
    assert Path(config["checkpoint_dir"], "latest.pth").is_file()
    generator = UNetGenerator(base_channels=2)
    discriminator = PatchDiscriminator(base_channels=2)
    optimizer_g = torch.optim.Adam(generator.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())
    checkpoint = load_training_checkpoint(
        Path(config["checkpoint_dir"], "latest.pth"),
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        torch.device("cpu"),
    )
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] >= 1
    report = evaluate(
        str(Path(config["checkpoint_dir"], "best_generator.pth")),
        clean_dir=str(clean),
        split_manifest=str(Path(config["data"]["split_manifest"])),
        output_dir=str(tmp_path / "held_out_report"),
    )
    assert report["split"] == "test"
    assert report["sample_count"] == 1
    assert "restored_ssim" in report["mean_metrics"]
    assert Path(tmp_path / "held_out_report" / "comparison_grid.png").is_file()
