import numpy as np
import pytest

from Processor import ImageProcessor


def test_turn_gray_uses_weighted_luminance():
    image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

    result = ImageProcessor.turn_gray(image)

    np.testing.assert_array_equal(result, np.array([[76, 149, 29]], dtype=np.uint8))


def test_turn_binary_applies_threshold_to_color_image():
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)

    result = ImageProcessor.turn_binary(image, threshold=127)

    np.testing.assert_array_equal(result, np.array([[0, 255]], dtype=np.uint8))


def test_histogram_counts_every_pixel():
    image = np.array([[0, 0], [128, 255]], dtype=np.uint8)

    histogram = ImageProcessor.get_histogram(image)

    assert histogram.sum() == image.size
    assert histogram[0] == 2
    assert histogram[128] == 1
    assert histogram[255] == 1


def test_resize_manual_uses_nearest_neighbor():
    image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    result = ImageProcessor.resize_manual(image, 2)

    np.testing.assert_array_equal(
        result,
        np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize("scale_factor", [0, -1])
def test_resize_manual_rejects_non_positive_scale(scale_factor):
    with pytest.raises(ValueError, match="greater than zero"):
        ImageProcessor.resize_manual(np.zeros((2, 2), dtype=np.uint8), scale_factor)


def test_add_images_saturates_at_255():
    first = np.array([[250, 10]], dtype=np.uint8)
    second = np.array([[20, 15]], dtype=np.uint8)

    result = ImageProcessor.add_images_manual(first, second)

    np.testing.assert_array_equal(result, np.array([[255, 25]], dtype=np.uint8))


def test_rotation_matches_clockwise_numpy_rotation():
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    result = ImageProcessor.rotation_image(image)

    np.testing.assert_array_equal(result, np.rot90(image, k=-1))


def test_histogram_equalization_preserves_constant_image():
    image = np.full((3, 3), 42, dtype=np.uint8)

    result = ImageProcessor.histogram_equalization_manual(image)

    np.testing.assert_array_equal(result, image)


def test_noise_is_repeatable_with_seed():
    image = np.full((8, 8), 127, dtype=np.uint8)

    first = ImageProcessor.add_salt_pepper_noise_manual(image, amount=0.5, seed=7)
    second = ImageProcessor.add_salt_pepper_noise_manual(image, amount=0.5, seed=7)

    np.testing.assert_array_equal(first, second)
