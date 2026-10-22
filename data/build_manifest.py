"""Build and report normalized DIV2K HR/LR pairs and source-level splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from data.div2k_dataset import build_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HR = ROOT / "DIV2K" / "DIV2K" / "DIV2K_train_HR"
DEFAULT_LR = ROOT / "DIV2K" / "DIV2K" / "DIV2K_train_LR_unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-dir", default=str(DEFAULT_HR))
    parser.add_argument("--lr-root", default=str(DEFAULT_LR))
    parser.add_argument("--scale", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=str(ROOT / "data" / "splits" / "div2k_train.json"))
    parser.add_argument("--report", default=str(ROOT / "outputs" / "dataset_pairing_report.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build_manifest(
        args.hr_dir,
        args.lr_root,
        args.manifest,
        args.report,
        scale=args.scale,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        force=args.force,
    )
    test_root = Path(args.lr_root).parent / "DIV2K_test_LR_unknown"
    test_hr = test_root.parent / "DIV2K_test_HR"
    test_summary = {
        "lr_counts": {
            f"X{scale}": len(
                [
                    path
                    for path in (test_root / f"X{scale}").rglob("*")
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                ]
            )
            if (test_root / f"X{scale}").is_dir()
            else 0
            for scale in (2, 3, 4)
        },
        "hr_directory_available": test_hr.is_dir(),
        "note": "LR-only test inputs cannot be scored with PSNR/SSIM without paired HR references.",
    }
    report_path = Path(args.report)
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    report_data["separate_test_set"] = test_summary
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    summary = {
        scale: {
            "hr_files": report["hr_files"],
            "lr_files": report["lr_files"],
            "matched_valid_pairs": report["matched_valid_pairs"],
            "unmatched_hr": len(report["unmatched_hr"]),
            "unmatched_lr": len(report["unmatched_lr"]),
            "duplicates": len(report["duplicate_hr_keys"]) + len(report["duplicate_lr_keys"]),
            "invalid_files": len(report["invalid_files"]),
            "excluded_pairs": len(report["excluded_pairs"]),
        }
        for scale, report in result["reports"].items()
    }
    selected_pairs = [pair for pair in result["manifest"]["pairs"] if pair["scale"] == args.scale]
    summary["selected_split_counts"] = {
        split: sum(pair["split"] == split for pair in selected_pairs) for split in ("train", "validation")
    }
    summary["separate_test_set"] = test_summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
