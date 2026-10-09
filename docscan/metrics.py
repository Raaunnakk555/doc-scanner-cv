"""Quantitative image-quality and detection metrics used for evaluation."""
import cv2
import numpy as np

from .geometry import order_points
from .preprocess import compute_histogram, to_gray


def sharpness(img):
    """Variance of the Laplacian - higher means sharper edges."""
    return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())


def rms_contrast(img):
    """Standard deviation of grey levels."""
    return float(to_gray(img).std())


def entropy(img):
    """Shannon entropy (bits) of the grey-level histogram."""
    hist = compute_histogram(to_gray(img)).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def quality_report(img):
    return {"sharpness": sharpness(img), "contrast": rms_contrast(img), "entropy": entropy(img)}


def corner_error(pred, truth):
    """Mean Euclidean distance (pixels) between matching ordered corners."""
    pred, truth = order_points(pred), order_points(truth)
    return float(np.mean(np.linalg.norm(pred - truth, axis=1)))
