"""Evaluate a trained checkpoint on explicit held-out reference pairs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset

from data.dataset import CleanImageDataset, PairedImageDataset, create_source_split, split_clean_dataset
from data.preprocessing import denormalize_image
from evaluation.metrics import image_metrics
from inference.restore import load_generator


@torch.inference_mode()
def evaluate(
    checkpoint: str,
    clean_dir: str | None = None,
    split_manifest: str | None = None,
    input_dir: str | None = None,
    target_dir: str | None = None,
    split: str = "test",
    output_dir: str = "outputs/evaluation",
    seed: int = 42,
) -> dict:
    model, device, config = load_generator(checkpoint)
    data_config = config.get("data", {}) if isinstance(config, dict) else {}
    image_size = int(data_config.get("image_size", 128))

    if clean_dir:
        manifest = split_manifest or data_config.get("split_manifest", "data/splits/source_split.json")
        selected = split_clean_dataset(
            clean_dir,
            manifest,
            int(config.get("seed", seed)),
            (
                float(data_config.get("train_fraction", 0.8)),
                float(data_config.get("validation_fraction", 0.1)),
                float(data_config.get("test_fraction", 0.1)),
            ),
        )[split]
        dataset = CleanImageDataset(selected, image_size, seed=seed + 200_000, training=False)
    elif input_dir and target_dir:
        dataset = PairedImageDataset(input_dir, target_dir, image_size)
        if split_manifest:
            source_paths = [pair[0] for pair in dataset.pairs]
            assignments = create_source_split(
                source_paths,
                split_manifest,
                seed=seed,
            )
            positions = {str(path.resolve()): index for index, path in enumerate(source_paths)}
            dataset = Subset(dataset, [positions[path] for path in assignments[split]])
    else:
        raise ValueError("Provide --clean-dir with a saved split manifest or explicit paired input/target dirs")
    if not len(dataset):
        raise ValueError(f"Evaluation split '{split}' is empty")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[float]] = {
        "degraded_ssim": [],
        "degraded_psnr": [],
        "restored_ssim": [],
        "restored_psnr": [],
    }
    comparisons: list[Image.Image] = []
    total_inference = 0.0
    for item in dataset:
        degraded = denormalize_image(item["input"])
        target = denormalize_image(item["target"])
        tensor = item["input"].unsqueeze(0).to(device)
        started = time.perf_counter()
        restored_tensor = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_inference += time.perf_counter() - started
        restored = denormalize_image(restored_tensor[0].cpu())
        baseline = image_metrics(np.asarray(target), np.asarray(degraded))
        result = image_metrics(np.asarray(target), np.asarray(restored))
        metrics["degraded_ssim"].append(baseline["ssim"])
        metrics["degraded_psnr"].append(baseline["psnr"])
        metrics["restored_ssim"].append(result["ssim"])
        metrics["restored_psnr"].append(result["psnr"])
        if len(comparisons) < 8:
            strip = Image.new("RGB", (target.width * 3, target.height))
            for column, image in enumerate((target, degraded, restored)):
                strip.paste(image, (column * target.width, 0))
            comparisons.append(strip)

    count = len(dataset)
    mean_metrics = {}
    for key, values in metrics.items():
        value = float(np.mean(values))
        mean_metrics[key] = value if math.isfinite(value) else "Infinity"
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split,
        "sample_count": count,
        "device": str(device),
        "mean_inference_seconds": total_inference / count,
        "mean_metrics": mean_metrics,
        "restored_psnr_improved": (
            mean_metrics["restored_psnr"] == "Infinity"
            or (
                mean_metrics["degraded_psnr"] != "Infinity"
                and mean_metrics["restored_psnr"] > mean_metrics["degraded_psnr"]
            )
        ),
        "ssim_target": 0.80,
        "ssim_target_met": float(np.mean(metrics["restored_ssim"])) >= 0.80,
        "protocol": "Sources split before corruption; checkpoint selection must use validation only.",
        "dataset_scope": "Synthetic paired results describe only this synthetic degradation distribution.",
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (comparisons[0].width, sum(image.height for image in comparisons)))
    offset = 0
    for comparison in comparisons:
        sheet.paste(comparison, (0, offset))
        offset += comparison.height
    sheet.save(output / "comparison_grid.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean-dir")
    parser.add_argument("--split-manifest")
    parser.add_argument("--input-dir")
    parser.add_argument("--target-dir")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = evaluate(
        args.checkpoint,
        clean_dir=args.clean_dir,
        split_manifest=args.split_manifest,
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
