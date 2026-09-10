import numpy as np
import pytest

from Image_Capture import load_image, save_to_image


def test_save_and_load_image_round_trip(tmp_path):
    image = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    output_path = tmp_path / "sample.png"

    save_to_image(image, output_path)
    loaded = load_image(output_path)

    np.testing.assert_array_equal(loaded, image)


def test_load_image_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Unable to read image"):
        load_image(tmp_path / "missing.png")
