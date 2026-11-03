from PIL import Image

from src.food11.data import CLASSES
from src.food11.validate_data import SPLITS, validate_dataset


def test_detects_cross_split_duplicate(tmp_path):
    for split_index, split in enumerate(SPLITS):
        for class_index, class_name in enumerate(CLASSES):
            folder = tmp_path / split / class_name
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (split_index + 1, class_index + 1, 17)).save(
                folder / "sample.png"
            )
    assert validate_dataset(tmp_path, min_per_class=1)

    source = tmp_path / "training" / CLASSES[0] / "sample.png"
    target = tmp_path / "evaluation" / CLASSES[0] / "sample.png"
    target.write_bytes(source.read_bytes())
    assert not validate_dataset(tmp_path, min_per_class=1)
