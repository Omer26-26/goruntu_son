# Image Processing Project

A desktop image processing application built with Python. The operations are implemented with NumPy and manual algorithm logic instead of ready-made image processing functions.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Dependencies

Runtime and test dependencies are version-bounded in [`requirements.txt`](requirements.txt).

## Available Operations

- Grayscale conversion and thresholding
- Contrast adjustment, histogram stretching, and histogram equalization
- RGB-HSV and RGB-YCbCr color conversions
- Image rotation, zooming, and cropping
- Salt and pepper noise generation
- Mean, median, and motion blur filters
- Sobel and Canny edge detection
- Morphological operations: erosion, dilation, opening, and closing
- Arithmetic operations with two images: addition, subtraction, multiplication, AND, OR, XOR
- Adaptive thresholding and double thresholding

## Files

- `main.py`: User interface and operation selection
- `Processor.py`: Image processing algorithms
- `Image_Capture.py`: Image loading and display helpers
- `tests/`: Deterministic pytest unit tests for algorithms and file I/O

## Tests

```bash
python -m pytest tests --verbose
```

The tests generate NumPy arrays and temporary files, so they do not depend on personal file paths or sample images.

## License

This project is licensed under the [MIT License](LICENSE).

