import unittest

import cv2
import numpy as np

from docscan.enhance import binarize, correct_illumination
from docscan.preprocess import (compute_histogram, equalize_histogram,
                                resize_for_processing, to_gray)


class TestHistogram(unittest.TestCase):
    def test_histogram_counts_sum_to_pixels(self):
        img = np.random.default_rng(1).integers(0, 256, (40, 50), dtype=np.uint8)
        hist = compute_histogram(img)
        self.assertEqual(hist.sum(), img.size)
        self.assertEqual(len(hist), 256)

    def test_equalisation_stretches_low_contrast(self):
        img = np.random.default_rng(2).integers(100, 140, (80, 80), dtype=np.uint8)
        eq = equalize_histogram(img)
        self.assertEqual(eq.min(), 0)
        self.assertEqual(eq.max(), 255)
        self.assertGreater(eq.std(), img.std() * 3)

    def test_equalisation_close_to_opencv(self):
        img = np.random.default_rng(3).integers(60, 200, (64, 64), dtype=np.uint8)
        diff = np.abs(equalize_histogram(img).astype(int) - cv2.equalizeHist(img).astype(int))
        self.assertLessEqual(diff.max(), 2)

    def test_flat_image_unchanged(self):
        img = np.full((30, 30), 90, np.uint8)
        np.testing.assert_array_equal(equalize_histogram(img), img)

    def test_requires_uint8(self):
        with self.assertRaises(ValueError):
            compute_histogram(np.zeros((5, 5), np.float32))


class TestHelpers(unittest.TestCase):
    def test_resize_keeps_small_images(self):
        img = np.zeros((100, 120, 3), np.uint8)
        out, scale = resize_for_processing(img, 1000)
        self.assertEqual(scale, 1.0)
        self.assertEqual(out.shape, img.shape)

    def test_resize_downscales_large_images(self):
        img = np.zeros((2000, 1000, 3), np.uint8)
        out, scale = resize_for_processing(img, 1000)
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(max(out.shape[:2]), 1000)

    def test_to_gray_shapes(self):
        self.assertEqual(to_gray(np.zeros((10, 10, 3), np.uint8)).ndim, 2)
        self.assertEqual(to_gray(np.zeros((10, 10), np.uint8)).ndim, 2)


class TestEnhance(unittest.TestCase):
    def test_illumination_correction_flattens_gradient(self):
        page = np.full((200, 300), 230, np.uint8)
        cv2.putText(page, "TEXT", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, 40, 3)
        ramp = np.linspace(0.4, 1.0, 300)[None, :]
        shaded = (page * ramp).astype(np.uint8)
        fixed = correct_illumination(shaded, 0.1)
        left, right = fixed[:, :40][:20].mean(), fixed[:, -40:][:20].mean()
        self.assertLess(abs(left - right), 15)                       # corrected
        self.assertGreater(abs(shaded[:20, :40].mean() - shaded[:20, -40:].mean()), 60)  # was not

    def test_binarize_is_binary(self):
        gray = np.random.default_rng(4).integers(0, 256, (60, 60), dtype=np.uint8)
        self.assertTrue(set(np.unique(binarize(gray))) <= {0, 255})


if __name__ == "__main__":
    unittest.main()
