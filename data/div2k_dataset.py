"""Robust paired DIV2K LR/HR discovery, validation, and scale-aware crops."""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from data.preprocessing import normalize_image

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def _image_key(path: Path, scale: int | None = None) -> str:
    stem = path.stem.casefold()
    if scale is not None:
        suffix = re.compile(rf"(?:[_-]?x{scale})$", re.IGNORECASE)
        stem = suffix.sub("", stem)
    return stem


def _index(paths: list[Path], scale: int | None = None) -> tuple[dict[str, Path], dict[str, list[str]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[_image_key(path, scale)].append(path)
    unique = {key: values[0] for key, values in grouped.items() if len(values) == 1}
    duplicates = {key: [str(path) for path in values] for key, values in grouped.items() if len(values) > 1}
    return unique, duplicates


def inspect_scale(
    hr_dir: str | Path,
    lr_dir: str | Path,
    scale: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair files by normalized source name and exclude invalid/ambiguous pairs."""
    if scale not in (2, 3, 4):
        raise ValueError(f"Unsupported DIV2K scale X{scale}; expected X2, X3, or X4")
    hr_paths = _files(Path(hr_dir))
    lr_paths = _files(Path(lr_dir))
    hr_by_key, duplicate_hr = _index(hr_paths)
    lr_by_key, duplicate_lr = _index(lr_paths, scale)
    hr_keys = set(hr_by_key) | set(duplicate_hr)
    lr_keys = set(lr_by_key) | set(duplicate_lr)
    invalid_files: list[dict[str, str]] = []
    excluded_pairs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for key in sorted(hr_keys & lr_keys):
        if key in duplicate_hr or key in duplicate_lr:
            excluded_pairs.append(
                {
                    "key": key,
                    "reason": "ambiguous_duplicate_key",
                    "hr_candidates": duplicate_hr.get(key, [str(hr_by_key[key])] if key in hr_by_key else []),
                    "lr_candidates": duplicate_lr.get(key, [str(lr_by_key[key])] if key in lr_by_key else []),
                }
            )
            continue
        hr_path, lr_path = hr_by_key[key], lr_by_key[key]
        try:
            with Image.open(hr_path) as image:
                hr_size = image.size
                image.verify()
        except (OSError, ValueError, RuntimeError) as exc:
            invalid_files.append({"path": str(hr_path), "error": str(exc)})
            excluded_pairs.append({"key": key, "reason": f"invalid_hr_image: {exc}"})
            continue
        try:
            with Image.open(lr_path) as image:
                lr_size = image.size
                image.verify()
        except (OSError, ValueError, RuntimeError) as exc:
            invalid_files.append({"path": str(lr_path), "error": str(exc)})
            excluded_pairs.append({"key": key, "reason": f"invalid_lr_image: {exc}"})
            continue
        expected_size = (lr_size[0] * scale, lr_size[1] * scale)
        if hr_size != expected_size:
            excluded_pairs.append(
                {
                    "key": key,
                    "reason": "scale_dimensions_mismatch",
                    "hr_size": list(hr_size),
                    "lr_size": list(lr_size),
                    "expected_hr_size": list(expected_size),
                }
            )
            continue
        pairs.append(
            {
                "image_id": key,
                "scale": scale,
                "hr_path": str(hr_path.resolve()),
                "lr_path": str(lr_path.resolve()),
                "hr_size": list(hr_size),
                "lr_size": list(lr_size),
            }
        )

    for key in sorted(set(duplicate_hr) - lr_keys):
        excluded_pairs.append({"key": key, "reason": "ambiguous_unmatched_hr", "hr_candidates": duplicate_hr[key]})
    for key in sorted(set(duplicate_lr) - hr_keys):
        excluded_pairs.append({"key": key, "reason": "ambiguous_unmatched_lr", "lr_candidates": duplicate_lr[key]})
    report = {
        "scale": f"X{scale}",
        "hr_files": len(hr_paths),
        "lr_files": len(lr_paths),
        "matched_valid_pairs": len(pairs),
        "unmatched_hr": [
            {"key": key, "path": str(path)} for key, path in sorted(hr_by_key.items()) if key not in lr_keys
        ],
        "unmatched_lr": [
            {"key": key, "path": str(path)} for key, path in sorted(lr_by_key.items()) if key not in hr_keys
        ],
        "duplicate_hr_keys": duplicate_hr,
        "duplicate_lr_keys": duplicate_lr,
        "invalid_files": invalid_files,
        "excluded_pairs": excluded_pairs,
    }
    LOGGER.info(
        "%s: HR=%d LR=%d valid_pairs=%d unmatched_HR=%d unmatched_LR=%d excluded=%d",
        report["scale"],
        report["hr_files"],
        report["lr_files"],
        report["matched_valid_pairs"],
        len(report["unmatched_hr"]),
        len(report["unmatched_lr"]),
        len(excluded_pairs),
    )
    return pairs, report


def build_manifest(
    hr_dir: str | Path,
    lr_root: str | Path,
    manifest_path: str | Path,
    report_path: str | Path,
    *,
    scale: int = 4,
    validation_fraction: float = 0.1,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    manifest_file, report_file = Path(manifest_path), Path(report_path)
    if manifest_file.exists() and not force:
        raise FileExistsError(f"Manifest exists; pass --force to rebuild: {manifest_file}")
    hr_path, lr_path = Path(hr_dir), Path(lr_root)
    all_pairs: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for current_scale in (2, 3, 4):
        folder = lr_path / f"X{current_scale}"
        if not folder.is_dir():
            reports[f"X{current_scale}"] = {
                "scale": f"X{current_scale}",
                "hr_files": len(_files(hr_path)),
                "lr_files": 0,
                "matched_valid_pairs": 0,
                "unmatched_hr": [],
                "unmatched_lr": [],
                "duplicate_hr_keys": {},
                "duplicate_lr_keys": {},
                "invalid_files": [],
                "excluded_pairs": [],
                "missing_directory": str(folder),
            }
            continue
        pairs, report = inspect_scale(hr_path, folder, current_scale)
        reports[f"X{current_scale}"] = report
        all_pairs.extend(pairs)
    selected = [pair for pair in all_pairs if pair["scale"] == scale]
    identities = sorted({pair["image_id"] for pair in selected})
    if len(identities) < 2:
        raise ValueError(f"Need at least two valid X{scale} source images, found {len(identities)}")
    random.Random(seed).shuffle(identities)
    validation_count = max(1, round(len(identities) * validation_fraction))
    if validation_count >= len(identities):
        validation_count = len(identities) - 1
    validation_ids = set(identities[:validation_count])
    for pair in all_pairs:
        pair["split"] = "validation" if pair["image_id"] in validation_ids else "train"
    manifest = {
        "format_version": 1,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "selected_scale": scale,
        "hr_root": str(hr_path.resolve()),
        "lr_root": str(lr_path.resolve()),
        "pairs": all_pairs,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_file.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return {"manifest": manifest, "reports": reports}


class DIV2KPairedDataset(Dataset[dict[str, torch.Tensor]]):
    """On-demand aligned LR/HR crop loader for one scale and one split."""

    def __init__(
        self,
        manifest_path: str | Path,
        scale: int = 4,
        split: str = "train",
        lr_crop_size: int = 64,
        seed: int = 42,
    ) -> None:
        if scale not in (2, 3, 4):
            raise ValueError(f"Unsupported scale X{scale}")
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if lr_crop_size < 1:
            raise ValueError("lr_crop_size must be positive")
        self.scale = scale
        self.split = split
        self.lr_crop_size = lr_crop_size
        self.seed = seed
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.pairs = [
            pair for pair in data.get("pairs", []) if pair["scale"] == scale and pair["split"] == split
        ]
        if not self.pairs:
            raise ValueError(f"No X{scale} pairs found in split '{split}'")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        try:
            with Image.open(pair["lr_path"]) as image:
                lr = ImageOps.exif_transpose(image).convert("RGB")
            with Image.open(pair["hr_path"]) as image:
                hr = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load DIV2K pair {pair['image_id']}: {exc}") from exc
        expected = (lr.width * self.scale, lr.height * self.scale)
        if hr.size != expected:
            raise ValueError(
                f"Pair {pair['image_id']} changed after manifest creation: LR={lr.size}, HR={hr.size}, "
                f"expected HR={expected}"
            )
        crop = min(self.lr_crop_size, lr.width, lr.height)
        if crop < self.lr_crop_size:
            LOGGER.warning("%s too small for crop %d; using crop %d", pair["image_id"], self.lr_crop_size, crop)
        if self.split == "train":
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            left = rng.randint(0, lr.width - crop)
            top = rng.randint(0, lr.height - crop)
        else:
            left = (lr.width - crop) // 2
            top = (lr.height - crop) // 2
        lr_box = (left, top, left + crop, top + crop)
        scale = self.scale
        hr_box = (left * scale, top * scale, (left + crop) * scale, (top + crop) * scale)
        lr, hr = lr.crop(lr_box), hr.crop(hr_box)
        if self.split == "train":
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
            if rng.random() < 0.5:
                lr, hr = ImageOps.mirror(lr), ImageOps.mirror(hr)
            rotations = rng.randrange(4)
            if rotations:
                angle = 90 * rotations
                lr, hr = lr.rotate(angle, expand=True), hr.rotate(angle, expand=True)
        return {
            "lr": normalize_image(lr),
            "hr": normalize_image(hr),
            "image_id": pair["image_id"],
            "lr_path": pair["lr_path"],
            "hr_path": pair["hr_path"],
        }
