"""Prepare class-organized, 128x128 versions of the Food-11 dataset.

Run from the repository root with:
    uv run python ./src/food11/data.py
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


CATEGORIES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
)
SPLITS = ("training", "evaluation", "validation")
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100


def category_from_filename(path: Path) -> str:
    """Return a class name from Food-11's ``<class>_<number>.jpg`` name."""
    try:
        class_id = int(path.stem.split("_", maxsplit=1)[0])
        return CATEGORIES[class_id]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected Food-11 filename: {path.name}") from exc


def prepare_dataset(raw_dir: Path, output_dir: Path, mini_dir: Path) -> None:
    """Resize raw images and write full and development-sized datasets."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset was not found: {raw_dir}")

    # Rebuilding from scratch prevents stale images after an input change.
    for target in (output_dir, mini_dir):
        if target.exists():
            shutil.rmtree(target)

    counts: Counter[tuple[str, str]] = Counter()
    for split in SPLITS:
        split_dir = raw_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split was not found: {split_dir}")

        for source in sorted(split_dir.iterdir()):
            if not source.is_file():
                continue
            category = category_from_filename(source)
            relative = Path(split) / category / source.name
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source) as image:
                image = ImageOps.fit(
                    image.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS
                )
                image.save(destination, format="JPEG", quality=90, optimize=True)

            key = (split, category)
            if counts[key] < MINI_LIMIT:
                mini_destination = mini_dir / relative
                mini_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, mini_destination)
            counts[key] += 1

    print(f"Prepared {sum(counts.values())} images in {output_dir}")
    print(f"Prepared {sum(min(count, MINI_LIMIT) for count in counts.values())} images in {mini_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/food11_raw"))
    parser.add_argument("--output", type=Path, default=Path("data/food11_processed"))
    parser.add_argument(
        "--mini", type=Path, default=Path("data/food11_processed_mini")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.raw, args.output, args.mini)
