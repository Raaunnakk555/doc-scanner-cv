"""Generate result figures for the report/README (requires matplotlib).

Run from the repo root:  python scripts/make_figures.py
"""
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from docscan.config import ScanConfig                      # noqa: E402
from docscan.evaluate import run_evaluation                # noqa: E402
from docscan.pipeline import ScanPipeline                  # noqa: E402
from docscan.preprocess import compute_histogram, equalize_histogram, to_gray  # noqa: E402
from docscan.synthetic import generate_document, make_scene  # noqa: E402

OUT = os.path.join("docs", "figures")
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im


def stages():
    rng = np.random.default_rng(7)
    scene, _ = make_scene(generate_document(rng), rng)
    res = ScanPipeline(ScanConfig(mode="color", paper="a4")).run(scene)
    gray = ScanPipeline(ScanConfig(mode="gray", paper="a4")).run(scene).enhanced
    bw = ScanPipeline(ScanConfig(mode="bw", paper="a4")).run(scene).enhanced
    panels = [("1. Input photo", rgb(scene), None), ("2. Canny edges", res.edges, "gray"),
              ("3. Detected quad", rgb(res.overlay), None), ("4. Rectified (homography)", rgb(res.rectified), None),
              ("5a. Enhanced: color", rgb(res.enhanced), None), ("5b. Enhanced: gray", gray, "gray"),
              ("5c. Enhanced: bw", bw, "gray")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, im, cmap) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap=cmap); ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pipeline_stages.png"), dpi=110); plt.close(fig)


def histograms():
    rng = np.random.default_rng(3)
    scene, _ = make_scene(generate_document(rng), rng)
    res = ScanPipeline(ScanConfig(mode="gray")).run(scene)
    before = to_gray(res.rectified)
    eq = equalize_histogram(before)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for col, (im, name) in enumerate([(before, "Rectified (before)"), (eq, "Histogram equalised")]):
        axes[0, col].imshow(im, cmap="gray"); axes[0, col].set_title(name); axes[0, col].axis("off")
        axes[1, col].bar(range(256), compute_histogram(im), width=1.0, color="#3b6fb6")
        axes[1, col].set_xlabel("grey level"); axes[1, col].set_ylabel("pixels")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "histograms.png"), dpi=110); plt.close(fig)


def evaluation_chart(eval_dir):
    rows = list(csv.DictReader(open(os.path.join(eval_dir, "evaluation.csv"))))
    contour = [float(r["corner_error_px"]) for r in rows]
    hough = [float(r["hough_only_error_px"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - 0.2, contour, 0.4, label="Contour method (primary)", color="#3b6fb6")
    ax.bar(x + 0.2, np.nan_to_num(hough, nan=0), 0.4, label="Hough-only (fallback ablation)", color="#e08a3c")
    ax.axhline(10, color="red", ls="--", lw=1); ax.text(len(rows) - 1, 11, "10 px success threshold", ha="right", color="red")
    ax.set_xlabel("synthetic scene"); ax.set_ylabel("mean corner error (px)")
    ax.set_title("Corner localisation error on 30 synthetic scenes"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "evaluation.png"), dpi=110); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    summary = run_evaluation(30, os.path.join("docs", "evaluation"), save_samples=0)
    print(summary)
    stages(); histograms(); evaluation_chart(os.path.join("docs", "evaluation"))
    print("figures written to", OUT)
