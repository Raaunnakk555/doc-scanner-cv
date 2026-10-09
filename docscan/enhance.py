"""Image enhancement (Unit 1): illumination correction, CLAHE, thresholding."""
import cv2
import numpy as np

from .preprocess import to_gray


def correct_illumination(gray, kernel_ratio=0.08):
    """Flat-field correction: divide by an estimate of the page background.

    Dilation removes dark text, a large median blur smooths the result into a
    background map, and dividing by it cancels shadows and uneven lighting.
    """
    k = int(min(gray.shape[:2]) * kernel_ratio) | 1       # force odd
    k = max(k, 3)
    background = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    background = cv2.medianBlur(background, k)
    return cv2.divide(gray, background, scale=255)


def apply_clahe(gray, clip=2.0, tiles=8):
    """Contrast Limited Adaptive Histogram Equalisation."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(gray)


def sharpen(gray, amount=0.6, sigma=1.2):
    """Unsharp masking: original + amount * (original - blurred)."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def binarize(gray, block=31, c=12):
    """Adaptive Gaussian threshold - robust to uneven illumination."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def enhance(img, config):
    """Apply the enhancement chain selected by config.mode."""
    if config.mode == "color":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = correct_illumination(l, config.illum_kernel_ratio)
        l = apply_clahe(l, config.clahe_clip, config.clahe_tiles)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = correct_illumination(to_gray(img), config.illum_kernel_ratio)
    if config.mode == "gray":
        return sharpen(apply_clahe(gray, config.clahe_clip, config.clahe_tiles))
    return binarize(gray, config.adaptive_block, config.adaptive_c)   # "bw"
