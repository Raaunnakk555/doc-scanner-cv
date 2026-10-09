# Problem Statement

## Problem
Photographs of paper documents taken with a phone are rarely usable as-is: the page is
skewed by perspective, surrounded by background clutter, and unevenly lit with shadows.
This makes text hard to read, hard to print and hard to feed into later processing such as
OCR. Manual cropping and correction is slow and inexact.

## Scope
The project is a **command-line document scanner** that takes a photo of a single,
roughly rectangular page and automatically:

1. locates the page boundary (edge detection, contours, Hough-transform fallback),
2. removes the perspective distortion (a homography estimated with the Direct Linear
   Transform, implemented from scratch), and
3. enhances the result (illumination correction, CLAHE, sharpening or binarisation),

and then reports quantitative quality and accuracy metrics.

**In scope:** one dominant rectangular page per image; JPG/PNG/BMP/TIFF input; colour,
grey-scale and black-and-white output; batch processing; accuracy evaluation on synthetic
scenes with known ground truth.

**Out of scope:** OCR, multi-page documents, non-rectangular or curved (book-spine)
pages, mobile/GUI front-ends, deep-learning models.

## Target Users
- Students and staff who need clean scans of notes, forms or whiteboards without a scanner.
- Computer-vision students who want a worked example of image formation, projective
  geometry and feature extraction applied to a real problem.
- Developers who need a lightweight, dependency-minimal pre-processing stage in front of OCR.

## High-Level Features
- Automatic page detection with a primary (contour) and a fallback (Hough) strategy.
- From-scratch normalised DLT homography and perspective rectification.
- Optional A4 / Letter aspect-ratio prior.
- Three enhancement modes: `color`, `gray`, `bw`.
- Single-file, batch and demo commands; per-image quality metrics (sharpness, contrast, entropy).
- Evaluation harness with ground-truth corner error and a detector ablation study.
- Validated configuration, robust error handling, file + console logging, 40 automated tests.

## Course Mapping (Computer Vision syllabus)
| Unit | Concepts used |
|------|---------------|
| 1 - Image formation & low-level processing | Perspective projection, convolution/Gaussian filtering, histogram equalisation (from scratch), CLAHE, restoration of uneven illumination |
| 2 - Multi-view geometry | Projective transformations, homography, DLT with Hartley normalisation, rectification |
| 3 - Feature extraction & segmentation | Canny edges, Hough transform, contour / region-based boundary detection |
