"""CSV loss and metric logger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = [
            "epoch",
            "global_step",
            "generator_loss",
            "discriminator_loss",
            "adversarial_loss",
            "reconstruction_loss",
            "validation_ssim",
            "validation_psnr",
            "elapsed_seconds",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.fields).writeheader()

    def log(self, values: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writerow({field: values.get(field, "") for field in self.fields})
