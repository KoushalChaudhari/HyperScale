"""Compare bicubic, generator-pretrain, and SRGAN outputs on validation pairs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from data.div2k_dataset import DIV2KPairedDataset
from data.preprocessing import denormalize_image
from evaluation.metrics import image_metrics
from inference.super_resolve import bicubic_upscale, load_sr_generator
from training.srgan_utils import ROOT, checkpoint_path, load_config, resolve_data_path, resolve_path


def _metrics(reference: Image.Image, prediction: Image.Image) -> dict[str, float]:
    if reference.size != prediction.size:
        raise ValueError(f"Image size mismatch: reference {reference.size}, output {prediction.size}")
    return image_metrics(np.asarray(reference.convert("RGB")), np.asarray(prediction.convert("RGB")))


@torch.inference_mode()
def evaluate_srgan(
    config: dict,
    *,
    pretrain_checkpoint: str | Path | None = None,
    srgan_checkpoint: str | Path | None = None,
    split: str = "validation",
    limit_images: int | None = None,
) -> dict:
    if split != "validation":
        raise ValueError("This project only reports validation metrics; no paired DIV2K test HR directory exists")
    output_dir = resolve_path(config, "output_dir") / "evaluation"
    manifest_path = resolve_data_path(config, "manifest")
    scale = int(config["data"]["scale"])
    pretrain_path = (
        Path(pretrain_checkpoint)
        if pretrain_checkpoint
        else checkpoint_path(config, "generator_pretrain_inference")
    )
    srgan_path = (
        Path(srgan_checkpoint)
        if srgan_checkpoint
        else checkpoint_path(config, "srgan_generator_inference")
    )
    pretrain_model, device, pretrain_info = load_sr_generator(pretrain_path, expected_scale=scale)
    srgan_model, srgan_device, srgan_info = load_sr_generator(srgan_path, expected_scale=scale)
    if device != srgan_device:
        srgan_model.to(device)
    dataset = DIV2KPairedDataset(
        manifest_path,
        scale=scale,
        split="validation",
        lr_crop_size=int(config["data"]["lr_crop_size"]),
        seed=int(config["seed"]),
    )
    count = min(len(dataset), limit_images) if limit_images else len(dataset)
    if count < 1:
        raise ValueError("No validation pairs available for evaluation")
    metric_names = ("bicubic", "pretrained_generator", "srgan")
    scores = {name: {"psnr": [], "ssim": []} for name in metric_names}
    timings = {name: 0.0 for name in ("pretrained_generator", "srgan")}
    comparisons: list[Image.Image] = []
    for index in tqdm(range(count), desc="Evaluating validation pairs"):
        sample = dataset[index]
        lr = denormalize_image(sample["lr"])
        hr = denormalize_image(sample["hr"])
        bicubic = bicubic_upscale(lr, scale)
        input_tensor = sample["lr"].unsqueeze(0).to(device)
        stage_outputs: dict[str, Image.Image] = {"bicubic": bicubic}
        for name, model in (("pretrained_generator", pretrain_model), ("srgan", srgan_model)):
            model.eval()
            start = time.perf_counter()
            output = model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[name] += time.perf_counter() - start
            stage_outputs[name] = denormalize_image(output[0].cpu())
        for name, image in stage_outputs.items():
            result = _metrics(hr, image)
            scores[name]["psnr"].append(result["psnr"])
            scores[name]["ssim"].append(result["ssim"])
        if len(comparisons) < 8:
            strip = Image.new("RGB", (hr.width * 4, hr.height))
            for column, image in enumerate((bicubic, stage_outputs["pretrained_generator"], stage_outputs["srgan"], hr)):
                strip.paste(image, (column * hr.width, 0))
            comparisons.append(strip)

    means = {
        name: {
            metric: (
                "Infinity"
                if math.isinf(float(np.mean(values)))
                else float(np.mean(values))
            )
            for metric, values in values_by_metric.items()
        }
        for name, values_by_metric in scores.items()
    }
    report = {
        "split": "validation",
        "scale": f"X{scale}",
        "pair_count": count,
        "mean_inference_seconds_per_image": {name: timings[name] / count for name in timings},
        "metrics": means,
        "checkpoints": {
            "pretrained_generator": str(pretrain_path.resolve()),
            "pretrained_epoch": pretrain_info.get("epoch"),
            "srgan_generator": str(srgan_path.resolve()),
            "srgan_epoch": srgan_info.get("epoch"),
        },
        "excluded_pairs": len(
            json.loads(resolve_data_path(config, "pairing_report").read_text(encoding="utf-8"))[f"X{scale}"][
                "excluded_pairs"
            ]
        ),
        "test_set_note": "100 LR-only test inputs per scale are present, but no test HR directory exists; test PSNR/SSIM are unavailable.",
        "degradation_note": "LR_unknown uses an undocumented degradation pipeline; metrics assume supplied LR and HR are spatially aligned.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "srgan_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if comparisons:
        grid = Image.new("RGB", (comparisons[0].width, sum(image.height for image in comparisons)))
        top = 0
        for comparison in comparisons:
            grid.paste(comparison, (0, top))
            top += comparison.height
        grid.save(output_dir / "srgan_validation_grid.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--pretrain-checkpoint")
    parser.add_argument("--srgan-checkpoint")
    parser.add_argument("--split", default="validation", choices=("validation",))
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()
    result = evaluate_srgan(
        load_config(args.config),
        pretrain_checkpoint=args.pretrain_checkpoint,
        srgan_checkpoint=args.srgan_checkpoint,
        split=args.split,
        limit_images=args.limit_images,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
