"""Low-level image processing (Unit 1): grayscale, filtering, histograms."""
import cv2
import numpy as np


def to_gray(img):
    """Convert BGR to grayscale; pass grayscale through unchanged."""
    return img.copy() if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_for_processing(img, max_dim):
    """Downscale so the longest side <= max_dim. Returns (image, scale)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale == 1.0:
        return img.copy(), 1.0
    small = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def gaussian_blur(gray, kernel=5):
    """Suppress sensor noise before edge detection (low-pass convolution)."""
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def compute_histogram(gray):
    """256-bin intensity histogram, implemented directly with NumPy."""
    if gray.dtype != np.uint8:
        raise ValueError("histogram expects a uint8 image")
    return np.bincount(gray.ravel(), minlength=256)


def equalize_histogram(gray):
    """Global histogram equalisation written from first principles.

    s_k = round( (CDF(k) - CDF_min) / (N - CDF_min) * 255 )
    """
    hist = compute_histogram(gray)
    cdf = hist.cumsum()
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    total = gray.size
    if total == cdf_min:                     # flat image: nothing to equalise
        return gray.copy()
    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]
