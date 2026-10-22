import pytest
import torch
from PIL import Image

from data.div2k_dataset import DIV2KPairedDataset, build_manifest, inspect_scale
from models.sr_discriminator import SRDiscriminator
from models.sr_generator import SRGenerator


def _write_pair(hr_dir, lr_dir, name, scale=4, lr_size=(24, 18), color=(90, 100, 110)):
    Image.new("RGB", (lr_size[0] * scale, lr_size[1] * scale), color).save(hr_dir / f"{name}.png")
    Image.new("RGB", lr_size, color).save(lr_dir / f"{name}x{scale}.PNG")


def test_actual_style_div2k_pairing_and_source_split(tmp_path):
    hr = tmp_path / "hr"
    lr = tmp_path / "lr"
    hr.mkdir()
    for scale in (2, 3, 4):
        (lr / f"X{scale}").mkdir(parents=True)
    for index in range(10):
        name = f"{index + 1:04d}"
        Image.new("RGB", (96, 72), (10 + index, 20, 30)).save(hr / f"{name}.png")
        for scale in (2, 3, 4):
            Image.new("RGB", (96 // scale, 72 // scale), (10 + index, 20, 30)).save(
                lr / f"X{scale}" / f"{name}x{scale}.PNG"
            )
    result = build_manifest(hr, lr, tmp_path / "manifest.json", tmp_path / "report.json", scale=4)
    assert [result["reports"][f"X{s}"]["matched_valid_pairs"] for s in (2, 3, 4)] == [10, 10, 10]
    selected = [pair for pair in result["manifest"]["pairs"] if pair["scale"] == 4]
    assert sum(pair["split"] == "train" for pair in selected) == 9
    assert sum(pair["split"] == "validation" for pair in selected) == 1
    split_by_id = {pair["image_id"]: pair["split"] for pair in result["manifest"]["pairs"]}
    assert all(split_by_id[pair["image_id"]] == pair["split"] for pair in result["manifest"]["pairs"])
    dataset = DIV2KPairedDataset(tmp_path / "manifest.json", scale=4, split="validation", lr_crop_size=12)
    batch = dataset[0]
    assert batch["lr"].shape == (3, 12, 12)
    assert batch["hr"].shape == (3, 48, 48)
    assert dataset[0]["hr_path"] == batch["hr_path"]


def test_pairing_reports_ambiguous_unmatched_and_wrong_dimensions(tmp_path):
    hr, lr = tmp_path / "hr", tmp_path / "lr"
    hr.mkdir()
    lr.mkdir()
    Image.new("RGB", (32, 32)).save(hr / "scene.png")
    Image.new("RGB", (32, 32)).save(hr / "SCENE.jpg")
    Image.new("RGB", (8, 8)).save(lr / "scene_x4.png")
    Image.new("RGB", (8, 8)).save(lr / "orphanX4.png")
    pairs, report = inspect_scale(hr, lr, 4)
    assert pairs == []
    assert report["duplicate_hr_keys"]
    assert report["unmatched_lr"]
    assert report["excluded_pairs"][0]["reason"] == "ambiguous_duplicate_key"

    (hr / "SCENE.jpg").unlink()
    Image.new("RGB", (31, 31)).save(hr / "scene.png")
    pairs, report = inspect_scale(hr, lr, 4)
    assert pairs == []
    assert any(item["reason"] == "scale_dimensions_mismatch" for item in report["excluded_pairs"])


def test_sr_generator_and_discriminator_x4_step_and_x3_is_explicitly_unsupported():
    generator = SRGenerator(scale=4, features=8, residual_blocks=1)
    discriminator = SRDiscriminator(features=8)
    low = torch.randn(2, 3, 16, 16)
    high = torch.randn(2, 3, 64, 64)
    generated = generator(low)
    assert generated.shape == high.shape
    logits = discriminator(generated)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    try:
        SRGenerator(scale=3, features=8, residual_blocks=1)
    except ValueError as exc:
        assert "X2 and X4" in str(exc)
    else:
        raise AssertionError("X3 must not be incorrectly supported")


def test_manifest_is_not_silently_overwritten(tmp_path):
    hr = tmp_path / "hr"
    lr = tmp_path / "lr"
    hr.mkdir()
    for scale in (2, 3, 4):
        (lr / f"X{scale}").mkdir(parents=True)
    for index in range(2):
        _write_pair(hr, lr / "X4", f"{index:04d}")
        Image.new("RGB", (48, 36)).save(lr / "X2" / f"{index:04d}x2.png")
        Image.new("RGB", (32, 24)).save(lr / "X3" / f"{index:04d}x3.png")
    manifest = tmp_path / "manifest.json"
    build_manifest(hr, lr, manifest, tmp_path / "report.json")
    with pytest.raises(FileExistsError):
        build_manifest(hr, lr, manifest, tmp_path / "report.json")
