"""Projective geometry (Unit 2): homography estimation with the DLT.

The Direct Linear Transform is implemented from scratch with NumPy (including
Hartley normalisation) so the maths is visible; OpenCV is only used for the
final pixel resampling.
"""
import cv2
import numpy as np


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()          # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _normalise(pts):
    """Hartley normalisation: zero mean, mean distance sqrt(2)."""
    centroid = pts.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(pts - centroid, axis=1))
    s = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([[s, 0, -s * centroid[0]],
                  [0, s, -s * centroid[1]],
                  [0, 0, 1]])
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ homog.T).T[:, :2], T


def compute_homography_dlt(src, dst):
    """Estimate the 3x3 homography H with dst ~ H * src from >= 4 point pairs."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape or len(src) < 4:
        raise ValueError("need the same number (>= 4) of source and destination points")

    src_n, T_src = _normalise(src)
    dst_n, T_dst = _normalise(dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(rows))
    H_n = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_n @ T_src      # undo the normalisation
    if abs(H[2, 2]) < 1e-12:
        raise ValueError("degenerate point configuration (collinear points?)")
    return H / H[2, 2]


def apply_homography(H, pts):
    """Map Nx2 points through H (with perspective divide)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    mapped = (H @ homog.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def output_size(quad, aspect=None):
    """Output size of the rectified page.

    By default the size comes from the quad edge lengths. A single image cannot
    reveal the true aspect ratio without camera intrinsics, so an optional
    `aspect` (width/height, e.g. A4 = 0.707) re-shapes it while keeping the area.
    """
    tl, tr, br, bl = quad
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 10 or height < 10:
        raise ValueError("detected quadrilateral is degenerate")
    if aspect:
        area = float(width * height)
        height = int(round(np.sqrt(area / aspect)))
        width = int(round(aspect * height))
    return width, height


def rectify(img, quad, aspect=None):
    """Warp the quadrilateral to a fronto-parallel rectangle.

    Returns (rectified_image, H) where H maps original -> rectified coordinates.
    """
    quad = order_points(quad)
    width, height = output_size(quad, aspect)
    target = np.array([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    H = compute_homography_dlt(quad, target)
    warped = cv2.warpPerspective(img, H, (width, height), flags=cv2.INTER_CUBIC)
    return warped, H
