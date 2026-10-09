"""Document boundary detection (Unit 3): contours first, Hough lines as fallback."""
from dataclasses import dataclass

import cv2
import numpy as np

from .edges import auto_canny, detect_lines, intersect
from .geometry import order_points
from .logger import get_logger
from .preprocess import gaussian_blur, resize_for_processing, to_gray


@dataclass
class Detection:
    quad: np.ndarray      # 4x2 float32 in ORIGINAL image coordinates (TL,TR,BR,BL)
    method: str           # "contour", "hough" or "full_frame"
    edges: np.ndarray     # Canny edge map (processing resolution)


def _quad_from_contours(edges, min_area, eps_ratio):
    """Largest convex 4-sided blob in the edge map, or None."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < min_area:
            break
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        for eps in (eps_ratio, 0.03, 0.04, 0.05):      # loosen until 4 vertices remain
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _edge_support(support_map, quad, samples=60):
    """Fraction of points along the quad's 4 sides that lie on (dilated) edges."""
    h, w = support_map.shape
    hits = total = 0
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        t = np.linspace(0.0, 1.0, samples)[:, None]
        pts = np.rint(a + t * (b - a)).astype(int)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        hits += int(support_map[pts[inside, 1], pts[inside, 0]].sum())
        total += samples
    return hits / total


def _edge_support(support_map, quad, samples=60):
    """Fraction of points along the quad's 4 sides that lie on (dilated) edges."""
    h, w = support_map.shape
    hits = 0
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        t = np.linspace(0.0, 1.0, samples)[:, None]
        pts = np.rint(a + t * (b - a)).astype(int)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        hits += int(support_map[pts[inside, 1], pts[inside, 0]].sum())
    return hits / (4.0 * samples)


def _quad_from_hough(edges, threshold_ratio, min_support=0.45):
    """Fallback: intersect the outermost strong ~horizontal and ~vertical Hough
    lines. Text rules inside the page are interior, so the page borders are the
    extremes. The candidate is rejected unless enough of its perimeter lies on
    real edges (a sanity gate so a bad guess never reaches the user)."""
    h, w = edges.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    horizontals, verticals = [], []
    for rho, theta in detect_lines(edges, threshold_ratio):
        if abs(np.sin(theta)) > 0.866:                  # within 30 deg of horizontal
            horizontals.append(((rho, theta), (rho - cx * np.cos(theta)) / np.sin(theta)))
        elif abs(np.cos(theta)) > 0.866:                # within 30 deg of vertical
            verticals.append(((rho, theta), (rho - cy * np.sin(theta)) / np.cos(theta)))

    def outer_pair(group, min_sep):
        if len(group) < 2:
            return None
        ordered = sorted(group, key=lambda g: g[1])
        (lo, lo_pos), (hi, hi_pos) = ordered[0], ordered[-1]
        return (lo, hi) if hi_pos - lo_pos > min_sep else None

    hpair, vpair = outer_pair(horizontals, 0.3 * h), outer_pair(verticals, 0.3 * w)
    if hpair is None or vpair is None:
        return None
    top, bottom = hpair
    left, right = vpair
    corners = [intersect(top, left), intersect(top, right),
               intersect(bottom, right), intersect(bottom, left)]
    if any(c is None for c in corners):
        return None
    quad = np.array(corners, dtype=np.float32)
    margin = 0.25                                       # tolerate slightly-off-frame corners
    if (quad[:, 0].min() < -margin * w or quad[:, 0].max() > (1 + margin) * w or
            quad[:, 1].min() < -margin * h or quad[:, 1].max() > (1 + margin) * h):
        return None
    support_map = (cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0).astype(np.uint8)
    return quad if _edge_support(support_map, quad) >= min_support else None


def detect_document(img, config, force_method=None):
    """Locate the document quadrilateral in a BGR image.

    force_method="hough" skips the contour detector (used for ablation studies).
    """
    log = get_logger()
    small, scale = resize_for_processing(img, config.max_process_dim)
    gray = gaussian_blur(to_gray(small), config.blur_kernel)
    edges = auto_canny(gray)
    min_area = config.min_area_ratio * gray.shape[0] * gray.shape[1]

    quad = None if force_method == "hough" else _quad_from_contours(
        edges, min_area, config.approx_epsilon_ratio)
    method = "contour"
    if quad is None:
        if force_method != "hough":
            log.warning("contour detection failed, trying Hough-line fallback")
        quad, method = _quad_from_hough(edges, config.hough_threshold_ratio), "hough"
    if quad is None:
        log.warning("no document found - using the full frame")
        h, w = small.shape[:2]
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        method = "full_frame"

    quad = order_points(quad / scale)                   # back to original resolution
    return Detection(quad=quad, method=method, edges=edges)
