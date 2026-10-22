"""Create clean geometric smoke-test images; they are not restoration-quality evidence."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import list_images


def _smoke_image(index: int, size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size), (35 + index * 17 % 80, 90, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, size - 24, size - 24), outline=(235, 220, 150), width=5)
    draw.ellipse((size // 4, size // 4, size * 3 // 4, size * 3 // 4), fill=(190, 80 + index * 11 % 100, 70))
    draw.line((0, index * 17 % size, size, (index * 17 + 80) % size), fill="white", width=3)
    draw.text((12, 12), f"SMOKE ONLY {index}", fill="white")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="Optional directory of user-owned clean RGB images")
    parser.add_argument("--output-dir", default="data/clean")
    parser.add_argument("--count", type=int, default=12, help="Number of smoke images if no source directory is supplied")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.source_dir:
        source_root = Path(args.source_dir)
        sources = list_images(source_root)
        if not sources:
            raise ValueError(f"No image files found in {args.source_dir}")
        copied = 0
        for source in sources:
            relative = source.relative_to(source_root)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
                copied += 1
        print(f"Prepared {len(sources)} user-provided clean images in {output} ({copied} copied).")
        return
    if args.count < 3:
        raise ValueError("At least 3 images are required to create disjoint source splits")
    for index in range(args.count):
        _smoke_image(index).save(output / f"smoke_only_{index:04d}.png")
    print(
        f"Created {args.count} geometric smoke-only images in {output}. "
        "They verify pipeline mechanics only and must not be used to claim restoration quality."
    )


if __name__ == "__main__":
    main()
