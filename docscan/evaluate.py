"""Batch evaluation on synthetic scenes with known ground truth."""
import csv
import os

import numpy as np

from .config import ScanConfig
from .detect import detect_document
from .io_utils import save_image
from .metrics import corner_error
from .pipeline import ScanPipeline
from .synthetic import generate_document, make_scene

SUCCESS_THRESHOLD_PX = 10.0


def run_evaluation(n, out_dir, config=None, seed0=0, save_samples=3):
    """Scan n synthetic scenes; write per-scene CSV and return a summary dict."""
    if n < 1:
        raise ValueError("n must be >= 1")
    os.makedirs(out_dir, exist_ok=True)
    pipeline = ScanPipeline(config or ScanConfig())
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        scene, gt = make_scene(generate_document(rng), rng)
        res = pipeline.run(scene)
        err = corner_error(res.quad, gt)
        hough = detect_document(scene, pipeline.config, force_method="hough")
        hough_err = corner_error(hough.quad, gt) if hough.method == "hough" else float("nan")
        rows.append({
            "scene": i, "method": res.method, "corner_error_px": round(err, 2),
            "success": int(err <= SUCCESS_THRESHOLD_PX),
            "hough_only_error_px": round(hough_err, 2),
            "contrast_before": round(res.metrics["before"]["contrast"], 2),
            "contrast_after": round(res.metrics["after"]["contrast"], 2),
            "entropy_before": round(res.metrics["before"]["entropy"], 3),
            "entropy_after": round(res.metrics["after"]["entropy"], 3),
        })
        if i < save_samples:
            save_image(os.path.join(out_dir, f"scene_{i}_input.png"), scene)
            save_image(os.path.join(out_dir, f"scene_{i}_detected.png"), res.overlay)
            save_image(os.path.join(out_dir, f"scene_{i}_scanned.png"), res.enhanced)

    csv_path = os.path.join(out_dir, "evaluation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errs = np.array([r["corner_error_px"] for r in rows])
    return {
        "scenes": n,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_corner_error_px": float(errs.mean()),
        "max_corner_error_px": float(errs.max()),
        "contour_method_share": float(np.mean([r["method"] == "contour" for r in rows])),
        "hough_only_success_rate": float(np.mean([
            (not np.isnan(r["hough_only_error_px"])) and r["hough_only_error_px"] <= SUCCESS_THRESHOLD_PX
            for r in rows])),
        "csv": csv_path,
    }
