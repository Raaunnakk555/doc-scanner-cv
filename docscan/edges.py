"""Edge and line detection (Unit 3): Canny and the Hough transform."""
import cv2
import numpy as np


def auto_canny(gray):
    """Canny edge detector with thresholds chosen automatically via Otsu.

    The high threshold is Otsu's optimal split of the histogram and the low
    threshold is half of it (Canny's recommended 1:2 ratio).
    """
    high, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(float(high), 30.0)
    return cv2.Canny(gray, 0.5 * high, high)


def detect_lines(edges, threshold_ratio=0.25):
    """Standard Hough transform. Returns a list of (rho, theta) sorted by votes."""
    h, w = edges.shape[:2]
    votes = max(20, int(threshold_ratio * min(h, w)))
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, votes)
    if lines is None:
        return []
    return [(float(r), float(t)) for r, t in lines[:, 0, :]]


def intersect(line_a, line_b):
    """Intersection of two lines given as (rho, theta). None if parallel."""
    (r1, t1), (r2, t2) = line_a, line_b
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    x, y = np.linalg.solve(A, np.array([r1, r2]))
    return float(x), float(y)
