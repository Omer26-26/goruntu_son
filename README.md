# Image Processing Project

A desktop image processing application built with Python. The operations are implemented with NumPy and manual algorithm logic instead of ready-made image processing functions.

## Usage

```bash
python main.py
```

## Dependencies

```bash
pip install numpy pillow matplotlib customtkinter
```

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
- `test.py`: Simple test file

