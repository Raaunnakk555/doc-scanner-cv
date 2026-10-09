import unittest

import cv2
import numpy as np

from docscan.geometry import (apply_homography, compute_homography_dlt,
                              order_points, output_size, rectify)


class TestHomography(unittest.TestCase):
    def setUp(self):
        self.src = np.array([[10, 20], [300, 30], [320, 400], [5, 380]], float)
        self.dst = np.array([[0, 0], [299, 0], [299, 399], [0, 399]], float)

    def test_dlt_maps_points_exactly(self):
        H = compute_homography_dlt(self.src, self.dst)
        np.testing.assert_allclose(apply_homography(H, self.src), self.dst, atol=1e-6)

    def test_dlt_matches_opencv(self):
        H = compute_homography_dlt(self.src, self.dst)
        H_cv = cv2.getPerspectiveTransform(self.src.astype(np.float32), self.dst.astype(np.float32))
        np.testing.assert_allclose(H, H_cv / H_cv[2, 2], rtol=1e-4, atol=1e-6)

    def test_identity_for_equal_points(self):
        H = compute_homography_dlt(self.src, self.src)
        np.testing.assert_allclose(H, np.eye(3), atol=1e-8)

    def test_overdetermined_with_noise_is_close(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 300, (30, 2))
        H_true = np.array([[1.1, 0.1, 5], [0.05, 0.9, -3], [1e-4, 2e-4, 1]])
        noisy = apply_homography(H_true, pts) + rng.normal(0, 0.1, (30, 2))
        H = compute_homography_dlt(pts, noisy)
        err = np.linalg.norm(apply_homography(H, pts) - apply_homography(H_true, pts), axis=1)
        self.assertLess(err.mean(), 0.5)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            compute_homography_dlt(self.src[:3], self.dst[:3])

    def test_collinear_points_raise(self):
        line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
        with self.assertRaises(Exception):
            H = compute_homography_dlt(line, self.dst)
            if not np.all(np.isfinite(H)):
                raise ValueError("non-finite homography")


class TestOrderingAndSize(unittest.TestCase):
    def test_order_points_any_permutation(self):
        ordered = np.array([[10, 10], [200, 20], [210, 300], [5, 290]], np.float32)
        for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
            np.testing.assert_allclose(order_points(ordered[perm]), ordered)

    def test_output_size_rectangle(self):
        quad = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], np.float32)
        self.assertEqual(output_size(quad), (100, 50))

    def test_output_size_aspect_keeps_area(self):
        quad = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], np.float32)
        w, h = output_size(quad, aspect=0.5)
        self.assertAlmostEqual(w / h, 0.5, delta=0.02)
        self.assertAlmostEqual(w * h, 5000, delta=150)

    def test_degenerate_quad_raises(self):
        quad = np.zeros((4, 2), np.float32)
        with self.assertRaises(ValueError):
            output_size(quad)

    def test_rectify_recovers_fronto_parallel_image(self):
        page = np.zeros((200, 300, 3), np.uint8)
        cv2.rectangle(page, (50, 50), (250, 150), (255, 255, 255), -1)
        quad = np.float32([[20, 30], [320, 10], [340, 260], [10, 240]])
        H0 = cv2.getPerspectiveTransform(np.float32([[0, 0], [299, 0], [299, 199], [0, 199]]), quad)
        photo = cv2.warpPerspective(page, H0, (360, 280))
        out, _ = rectify(photo, quad)
        self.assertEqual(out.shape[2], 3)
        # centre of the white rectangle must be white after rectification
        h, w = out.shape[:2]
        self.assertGreater(out[h // 2, w // 2].mean(), 200)


if __name__ == "__main__":
    unittest.main()
