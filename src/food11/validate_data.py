"""Validate the processed Food-11 dataset before training or release."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

try:
    from .data import CLASSES
except ImportError:  # Direct script execution from the repo root.
    from data import CLASSES


SPLITS = ("training", "validation", "evaluation")


def validate_dataset(root: Path, min_per_class: int = 50, tau: float = 0.05) -> bool:
    """Print each contract check and return whether every check passed."""
    failures: dict[str, list[str]] = {
        "class folders": [],
        "image format": [],
        "minimum images": [],
        "split duplicates": [],
        "class balance": [],
    }
    counts: dict[str, Counter[str]] = {}
    seen_hashes: dict[str, tuple[str, Path]] = {}

    for split in SPLITS:
        split_dir = root / split
        actual = (
            {path.name for path in split_dir.iterdir() if path.is_dir()}
            if split_dir.is_dir()
            else set()
        )
        expected = set(CLASSES)
        if actual != expected:
            failures["class folders"].append(
                f"{split}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        counts[split] = Counter()
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                failures["minimum images"].append(
                    f"{split}/{class_name}: 0 < {min_per_class}"
                )
                continue
            files = sorted(path for path in class_dir.iterdir() if path.is_file())
            counts[split][class_name] = len(files)
            if len(files) < min_per_class:
                failures["minimum images"].append(
                    f"{split}/{class_name}: {len(files)} < {min_per_class}"
                )
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (128, 128):
                            failures["image format"].append(
                                f"{path}: mode={image.mode}, size={image.size}"
                            )
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    failures["image format"].append(f"{path}: {exc}")
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                previous = seen_hashes.get(digest)
                if previous is not None and previous[0] != split:
                    failures["split duplicates"].append(
                        f"{path} duplicates {previous[1]}"
                    )
                else:
                    seen_hashes[digest] = (split, path)

    train_total = sum(counts["training"].values())
    val_total = sum(counts["validation"].values())
    if not train_total or not val_total:
        failures["class balance"].append("training or validation split is empty")
    else:
        for class_name in CLASSES:
            train_share = counts["training"][class_name] / train_total
            val_share = counts["validation"][class_name] / val_total
            if abs(train_share - val_share) > tau:
                failures["class balance"].append(
                    f"{class_name}: training={train_share:.3f}, validation={val_share:.3f}"
                )

    for check, issues in failures.items():
        print(f"{'FAIL' if issues else 'PASS'} {check}")
        for issue in issues[:5]:
            print(f"  {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
    return not any(failures.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    if args.min_per_class < 0 or not 0 <= args.tau <= 1:
        parser.error(
            "min-per-class must be nonnegative and tau must be between 0 and 1"
        )
    raise SystemExit(
        0 if validate_dataset(args.dataset, args.min_per_class, args.tau) else 1
    )


if __name__ == "__main__":
    main()
