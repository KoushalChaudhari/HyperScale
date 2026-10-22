import random

import numpy as np
from PIL import Image

from data.dataset import CleanImageDataset, PairedImageDataset, create_source_split
from data.degradation import degrade_image
from data.preprocessing import denormalize_image, normalize_image


def test_normalization_round_trip_and_degradation():
    source = Image.fromarray(np.full((128, 128, 3), 128, dtype=np.uint8))
    tensor = normalize_image(source)
    assert tensor.shape == (3, 128, 128)
    assert tensor.min() >= -1 and tensor.max() <= 1
    restored = denormalize_image(tensor)
    assert np.asarray(restored).shape == (128, 128, 3)
    assert degrade_image(source, random.Random(4)).size == source.size


def test_paired_dataset_and_disjoint_source_split(tmp_path):
    clean = tmp_path / "clean"
    degraded = tmp_path / "degraded"
    targets = tmp_path / "targets"
    for folder in (clean, degraded, targets):
        folder.mkdir()
    for index in range(10):
        image = Image.new("RGB", (128, 128), (index * 10, 50, 90))
        image.save(clean / f"{index}.png")
        image.save(degraded / f"{index}.png")
        image.save(targets / f"{index}.png")
    manifest = create_source_split(list(clean.glob("*.png")), tmp_path / "split.json", 7)
    assert not (set(manifest["train"]) & set(manifest["validation"]))
    assert not (set(manifest["train"]) & set(manifest["test"]))
    assert not (set(manifest["validation"]) & set(manifest["test"]))
    dataset = PairedImageDataset(degraded, targets, 128)
    assert len(dataset) == 10
    assert dataset[0]["input"].shape == (3, 128, 128)
    (targets / "missing.png").write_bytes((degraded / "0.png").read_bytes())
    try:
        PairedImageDataset(degraded, targets, 128)
    except ValueError as exc:
        assert "Unmatched paired filenames" in str(exc)
    else:
        raise AssertionError("Missing pairs must be reported")


def test_clean_dataset_corruption_reproducible_for_evaluation(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (128, 128), (90, 110, 130)).save(image_path)
    dataset = CleanImageDataset([image_path], training=False)
    first = dataset[0]
    second = dataset[0]
    assert (first["input"] == second["input"]).all()
    assert (first["target"] == second["target"]).all()
