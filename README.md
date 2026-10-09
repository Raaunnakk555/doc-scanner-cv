# Document Scanner & Enhancer (`docscan`)

**A Computer Vision project by Raunak Rustagi**

A command-line document scanner built using classical computer vision techniques. Given a photograph of a paper page taken at an angle, the system detects the page boundaries, corrects perspective distortion, and enhances the image to improve readability.

The project demonstrates practical applications of image processing, edge detection, contour analysis, projective geometry, homography estimation, and histogram-based enhancement using Python, NumPy, and OpenCV.

## Overview

```text
Input photograph
       |
       v
Image validation and preprocessing
       |
       v
Canny edge detection
       |
       v
Page boundary detection
(Contours + Hough fallback)
       |
       v
Homography estimation using DLT
       |
       v
Perspective correction
       |
       v
Image enhancement
       |
       v
Save scanned image and quality metrics
```

**Pipeline visualization:** [View pipeline stages](https://github.com/Raaunnakk555/doc-scanner-cv/blob/main/docs/figures/pipeline_stages.png)

## Features

### 1. Document Detection
- Gaussian filtering to reduce image noise.
- Otsu-based threshold selection for Canny edge detection.
- Contour-based detection of the document's four corners.
- Hough-transform fallback when contour detection cannot find a suitable page boundary.

### 2. Perspective Rectification
- Corner ordering and quadrilateral validation.
- Homography estimation using the Direct Linear Transform (DLT).
- Singular Value Decomposition (SVD) for solving the homography system.
- Perspective warping to produce a front-facing document image.
- Optional A4 and Letter paper aspect-ratio settings.

### 3. Image Enhancement
- Illumination correction to reduce uneven lighting and shadows.
- Histogram equalisation implemented using NumPy.
- Contrast Limited Adaptive Histogram Equalisation (CLAHE).
- Image sharpening.
- Adaptive thresholding for black-and-white output.

Available enhancement modes:
- `color`
- `gray`
- `bw`

### 4. Additional Capabilities
- Scan a single image.
- Process a folder of images in batch mode.
- Generate and scan synthetic document images.
- Evaluate detection accuracy using known ground-truth corners.
- Calculate image sharpness, contrast, entropy, and corner localisation error.
- Save logs, debug images, and evaluation results.
- Validate input files and handle processing errors.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application logic and command-line interface |
| NumPy | Numerical computations and DLT implementation |
| OpenCV | Image processing, edge detection, contours, and perspective warping |
| unittest | Automated unit and integration testing |
| Matplotlib | Optional generation of project diagrams and figures |

## Project Structure

```text
doc-scanner-cv/
├── main.py
├── docscan/
│   ├── config.py
│   ├── logger.py
│   ├── io_utils.py
│   ├── preprocess.py
│   ├── edges.py
│   ├── detect.py
│   ├── geometry.py
│   ├── enhance.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── synthetic.py
│   └── evaluate.py
├── tests/
├── scripts/
├── docs/
├── samples/
├── statement.md
├── requirements.txt
└── README.md
```

**Main components**

- `main.py` — command-line entry point for scanning, batch processing, demos, and evaluation.
- `detect.py` — document boundary detection.
- `geometry.py` — homography estimation and perspective rectification.
- `enhance.py` — illumination correction and image enhancement.
- `preprocess.py` — image preprocessing and histogram operations.
- `pipeline.py` — coordinates the complete scanning workflow.
- `metrics.py` — computes image quality metrics.
- `synthetic.py` and `evaluate.py` — generate synthetic scenes and evaluate detection accuracy.
- `tests/` — automated tests for geometry, preprocessing, detection, and pipeline behaviour.

## Installation

The project can be run on Windows, macOS, or Linux using a terminal.

### 1. Clone the repository

```bash
git clone https://github.com/Raaunnakk555/doc-scanner-cv.git
cd doc-scanner-cv
```

### 2. Check Python

Python 3.9 or later is intended by the project documentation. The commands below use the Python launcher on Windows.

```powershell
py --version
```

### 3. Install dependencies

```powershell
py -m pip install -r requirements.txt
```

## Usage

Run these commands from the project's root directory.

### Run the synthetic demo

```powershell
py main.py demo -o outputs/demo
```

### Scan a single photograph

```powershell
py main.py scan "path/to/photo.jpg" -o outputs/scan --mode bw --paper a4 --debug
```

Replace `path/to/photo.jpg` with the actual path to your image.

### Scan a folder of images

```powershell
py main.py batch "path/to/photos" -o outputs/batch
```

### Evaluate detection accuracy

```powershell
py main.py evaluate -n 30 -o outputs/eval
```

### View command help

```powershell
py main.py --help
py main.py scan --help
```

The generated images and evaluation files are saved in the selected output directories. Logging is configured by the application and can include `logs/docscan.log`.

**Exit codes:** `0` indicates success, `1` indicates an error, and `2` indicates a batch operation that finished with skipped files.

## Testing

Run the automated test suite:

```powershell
py -m unittest discover -s tests -t . -v
```

The recorded test run completed **40 tests successfully**.

The tests cover:

- Homography estimation and point mapping.
- Agreement with OpenCV's perspective-transform implementation.
- Corner ordering and degenerate geometry.
- Histogram equalisation and illumination correction.
- Image resizing and preprocessing.
- Document detection and fallback behaviour.
- Enhancement modes and contrast improvement.
- Input validation and command-line behaviour.

## Results

The project was evaluated on 30 synthetic scenes containing perspective distortion, lighting variation, and simulated image noise.

| Metric | Recorded result |
|---|---:|
| Detection success (corner error ≤ 10 pixels) | 100% |
| Mean corner error | 0.74 pixels |
| Maximum corner error | 1.13 pixels |
| Hough-only fallback success | 93% |
| Automated tests passed | 40 |

**Evaluation visualization:** [View evaluation results](https://github.com/Raaunnakk555/doc-scanner-cv/blob/main/docs/figures/evaluation.png)

These results describe the recorded synthetic evaluation. They should not be interpreted as guaranteed accuracy on every real-world photograph. Performance can vary with image quality, lighting, page visibility, and background complexity.

## Limitations

- Curved pages are not currently supported.
- Multiple documents in one image are outside the current scope.
- Heavily occluded or poorly visible page boundaries can be difficult to detect.
- The true paper aspect ratio cannot always be inferred from a single perspective-distorted image.
- Synthetic evaluation results may not fully represent real-world performance.
- The current implementation does not perform OCR or text recognition.

## Future Improvements

- Evaluate the system on a larger collection of real document photographs.
- Improve corner localisation and fallback detection.
- Add support for curved-page correction.
- Support multiple documents in one photograph.
- Integrate OCR as an optional post-processing stage.
- Extend robustness testing for blur, noise, and challenging lighting.

## Documentation

- [`statement.md`](https://github.com/Raaunnakk555/doc-scanner-cv/blob/main/statement.md) — project problem statement and scope.
- [`docs/figures/`](https://github.com/Raaunnakk555/doc-scanner-cv/tree/main/docs/figures) — pipeline and evaluation figures.
- [`docs/diagrams/`](https://github.com/Raaunnakk555/doc-scanner-cv/tree/main/docs/diagrams) — system design diagrams.
- [`docs/evaluation/`](https://github.com/Raaunnakk555/doc-scanner-cv/tree/main/docs/evaluation) — evaluation outputs, if present.

## Author

**Raunak Rustagi**  
Computer Vision Project  
VIT Bhopal University

GitHub: [Raaunnakk555](https://github.com/Raaunnakk555)

## License

Developed for educational and academic purposes.
