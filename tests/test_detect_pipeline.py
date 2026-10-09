import os
import tempfile
import unittest

import numpy as np

from docscan.config import ScanConfig
from docscan.detect import _quad_from_hough, detect_document
from docscan.edges import auto_canny, detect_lines, intersect
from docscan.io_utils import ImageLoadError, list_images, load_image, save_image
from docscan.metrics import corner_error, entropy
from docscan.pipeline import ScanPipeline
from docscan.preprocess import gaussian_blur, to_gray
from docscan.synthetic import generate_document, make_scene


def scene(seed=0):
    rng = np.random.default_rng(seed)
    return make_scene(generate_document(rng), rng)


class TestDetection(unittest.TestCase):
    def test_contour_detection_accuracy(self):
        for seed in range(5):
            img, gt = scene(seed)
            det = detect_document(img, ScanConfig())
            self.assertEqual(det.method, "contour")
            self.assertLess(corner_error(det.quad, gt), 5.0)

    def test_hough_fallback_accuracy(self):
        for seed in (1, 2, 3):
            img, gt = scene(seed)
            det = detect_document(img, ScanConfig(), force_method="hough")
            self.assertEqual(det.method, "hough")
            self.assertLess(corner_error(det.quad, gt), 10.0)

    def test_hough_rejects_weak_candidates(self):
        noise = np.random.default_rng(0).integers(0, 255, (300, 400), dtype=np.uint8)
        edges = auto_canny(gaussian_blur(noise))
        self.assertIsNone(_quad_from_hough(edges, 0.15))

    def test_blank_image_falls_back_to_full_frame(self):
        blank = np.full((300, 400, 3), 128, np.uint8)
        det = detect_document(blank, ScanConfig())
        self.assertEqual(det.method, "full_frame")

    def test_line_intersection(self):
        # x = 10 (rho=10, theta=0) and y = 20 (rho=20, theta=pi/2)
        x, y = intersect((10, 0.0), (20, np.pi / 2))
        self.assertAlmostEqual(x, 10, places=5)
        self.assertAlmostEqual(y, 20, places=5)
        self.assertIsNone(intersect((10, 0.0), (30, 0.0)))     # parallel

    def test_hough_finds_strong_lines(self):
        edges = np.zeros((200, 200), np.uint8)
        edges[100, :] = 255
        lines = detect_lines(edges, 0.5)
        self.assertTrue(any(abs(t - np.pi / 2) < 0.05 for _, t in lines))


class TestPipeline(unittest.TestCase):
    def test_all_modes_run(self):
        img, _ = scene(2)
        for mode in ("color", "gray", "bw"):
            res = ScanPipeline(ScanConfig(mode=mode)).run(img)
            self.assertEqual(res.enhanced.ndim, 3 if mode == "color" else 2)
            self.assertEqual(res.enhanced.dtype, np.uint8)

    def test_enhancement_raises_contrast(self):
        img, _ = scene(3)
        res = ScanPipeline(ScanConfig(mode="gray")).run(img)
        self.assertGreater(res.metrics["after"]["contrast"], res.metrics["before"]["contrast"])

    def test_grayscale_input_accepted(self):
        img, _ = scene(4)
        res = ScanPipeline().run(to_gray(img))
        self.assertEqual(res.method, "contour")

    def test_paper_aspect_applied(self):
        img, _ = scene(5)
        res = ScanPipeline(ScanConfig(paper="a4")).run(img)
        h, w = res.rectified.shape[:2]
        self.assertAlmostEqual(w / h, 210 / 297, delta=0.02)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ImageLoadError):
            ScanPipeline().run(np.zeros((10, 10, 3), np.uint8))   # too small
        with self.assertRaises(ImageLoadError):
            ScanPipeline().run("not an image")


class TestConfigAndIO(unittest.TestCase):
    def test_bad_config_values(self):
        for kwargs in ({"mode": "sepia"}, {"blur_kernel": 4}, {"min_area_ratio": 1.5},
                       {"paper": "a0"}, {"adaptive_block": 10}):
            with self.assertRaises(ValueError):
                ScanConfig(**kwargs)

    def test_load_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ImageLoadError):
                load_image(os.path.join(d, "missing.png"))
            bad_ext = os.path.join(d, "x.txt")
            with open(bad_ext, "w") as f:
                f.write("hi")
            with self.assertRaises(ImageLoadError):
                load_image(bad_ext)
            corrupt = os.path.join(d, "bad.png")
            with open(corrupt, "wb") as f:
                f.write(b"not a png")
            with self.assertRaises(ImageLoadError):
                load_image(corrupt)

    def test_save_load_roundtrip_and_listing(self):
        img, _ = scene(0)
        with tempfile.TemporaryDirectory() as d:
            path = save_image(os.path.join(d, "sub", "a.png"), img)
            np.testing.assert_array_equal(load_image(path), img)
            self.assertEqual(len(list_images(os.path.join(d, "sub"))), 1)
            with self.assertRaises(ImageLoadError):
                list_images(os.path.join(d, "nope"))

    def test_entropy_bounds(self):
        flat = np.full((50, 50), 7, np.uint8)
        self.assertEqual(entropy(flat), 0.0)


if __name__ == "__main__":
    unittest.main()
