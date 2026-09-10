import cv2

def bgr_to_rgb(image):
    """Convert an OpenCV BGR image to RGB."""
    return image[:, :, ::-1]

def rgb_to_bgr(image):
    """Convert an RGB image to OpenCV BGR."""
    return image[:, :, ::-1]

def load_image(path):
    """Load an image from any path-like object and return RGB data."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return bgr_to_rgb(image)

def save_to_image(image, path):
    """Save RGB image data to disk."""
    output_image = rgb_to_bgr(image)
    if not cv2.imwrite(str(path), output_image):
        raise OSError(f"Unable to write image: {path}")

def show_image(image, title="Image"):
    """Display an image in an OpenCV window until a key is pressed."""

    display_image = rgb_to_bgr(image) if image.ndim == 3 else image
    cv2.imshow(title, display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
