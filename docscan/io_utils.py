"""Image loading, saving and input validation."""
import os

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_SIDE = 50


class ImageLoadError(Exception):
    """Raised when an input image is missing, unsupported or unreadable."""


def validate_image_array(img):
    """Check that an in-memory array looks like a usable BGR/gray image."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ImageLoadError("image must be a uint8 numpy array")
    if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] != 3):
        raise ImageLoadError("image must be grayscale (H,W) or BGR (H,W,3)")
    if min(img.shape[:2]) < MIN_SIDE:
        raise ImageLoadError(f"image too small (min side {MIN_SIDE}px)")
    return img


def load_image(path):
    """Load an image from disk as a BGR array, with clear errors."""
    if not os.path.isfile(path):
        raise ImageLoadError(f"file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"unsupported format '{ext}'. Use one of {sorted(SUPPORTED_EXTENSIONS)}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"could not decode image (corrupt file?): {path}")
    return validate_image_array(img)


def save_image(path, img):
    """Write an image, creating parent folders as needed."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    if not cv2.imwrite(path, img):
        raise IOError(f"failed to write image: {path}")
    return path


def list_images(folder):
    """Return sorted paths of supported images inside a folder."""
    if not os.path.isdir(folder):
        raise ImageLoadError(f"folder not found: {folder}")
    names = sorted(os.listdir(folder))
    return [os.path.join(folder, n) for n in names
            if os.path.splitext(n)[1].lower() in SUPPORTED_EXTENSIONS]
