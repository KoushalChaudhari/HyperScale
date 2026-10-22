"""Clean-image and paired-image datasets with source-level split manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .degradation import degrade_image
from .preprocessing import load_rgb_image, normalize_image, prepare_training_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def create_source_split(
    paths: list[Path],
    manifest_path: str | Path,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Persist a deterministic source-level split; existing manifests are reused."""
    manifest = Path(manifest_path)
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        sources = {str(path.resolve()) for path in paths}
        split_names = ("train", "validation", "test")
        if any(not isinstance(data.get(key), list) for key in split_names):
            raise ValueError("Existing split manifest is missing train/validation/test lists")
        listed = [item for key in split_names for item in data[key]]
        if set(listed) != sources or len(listed) != len(sources):
            raise ValueError("Existing split manifest does not match current source images; choose a new manifest path")
        return data
    if len(fractions) != 3 or any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test fractions must be nonnegative and sum to 1")
    if len(paths) < 3:
        raise ValueError("At least three clean source images are required for disjoint train/validation/test splits")
    ordered = sorted(str(path.resolve()) for path in paths)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_train = max(1, int(n * fractions[0]))
    n_val = max(1, int(n * fractions[1]))
    if n_train + n_val >= n:
        n_train, n_val = n - 2, 1
    data = {
        "train": ordered[:n_train],
        "validation": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
        "seed": seed,
        "fractions": list(fractions),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


class CleanImageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        paths: list[str | Path],
        image_size: int = 128,
        seed: int = 42,
        training: bool = True,
        degradation_probabilities: dict[str, float] | None = None,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.image_size = image_size
        self.seed = seed
        self.training = training
        self.degradation_probabilities = degradation_probabilities
        if not self.paths:
            raise ValueError("Clean-image dataset is empty")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target = prepare_training_image(load_rgb_image(self.paths[index]), self.image_size)
        if self.training:
            if random.random() < 0.5:
                target = ImageOps.mirror(target)
            if random.random() < 0.5:
                target = ImageOps.flip(target)
            rng = random.Random(random.randrange(2**32))
        else:
            rng = random.Random(self.seed + index)
        degraded = degrade_image(target, rng=rng, probabilities=self.degradation_probabilities)
        return {"input": normalize_image(degraded), "target": normalize_image(target)}


class PairedImageDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        image_size: int = 128,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        inputs = {path.relative_to(self.input_dir).as_posix(): path for path in list_images(self.input_dir)}
        targets = {path.relative_to(self.target_dir).as_posix(): path for path in list_images(self.target_dir)}
        missing_targets = sorted(inputs.keys() - targets.keys())
        missing_inputs = sorted(targets.keys() - inputs.keys())
        if missing_targets or missing_inputs:
            raise ValueError(
                f"Unmatched paired filenames; missing targets: {missing_targets[:5]}, "
                f"missing inputs: {missing_inputs[:5]}"
            )
        self.pairs = [(inputs[name], targets[name]) for name in sorted(inputs)]
        self.image_size = image_size
        if not self.pairs:
            raise ValueError("Paired dataset is empty")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_path, target_path = self.pairs[index]
        degraded = load_rgb_image(source_path)
        target = load_rgb_image(target_path)
        if degraded.size != target.size:
            raise ValueError(f"Paired image dimensions do not match: {source_path.name}")
        return {
            "input": normalize_image(prepare_training_image(degraded, self.image_size)),
            "target": normalize_image(prepare_training_image(target, self.image_size)),
        }


def split_clean_dataset(
    clean_dir: str | Path,
    manifest_path: str | Path,
    seed: int,
    fractions: tuple[float, float, float],
) -> dict[str, list[str]]:
    return create_source_split(list_images(clean_dir), manifest_path, seed, fractions)
