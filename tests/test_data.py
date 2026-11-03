from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("3_120.jpg") == "Egg"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


def test_invalid_category():
    try:
        category_from_filename("11_0.jpg")
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range category must fail")


def test_resize_image_shape_and_mode():
    image = Image.new("L", (512, 384))
    result = resize_image(image)
    assert result.size == (128, 128)
    assert result.mode == "RGB"


def test_eleven_classes():
    assert len(CLASSES) == 11
