"""Generate the design diagrams used in the report (requires matplotlib).

Run from the repo root:  python scripts/make_diagrams.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

OUT = os.path.join("docs", "diagrams")
BLUE, GREEN, ORANGE, GREY, RED = "#dbe9ff", "#dff3e3", "#ffe9c9", "#eeeeee", "#fde0e0"


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    return fig, ax


def box(ax, x, y, w, h, text, fc=BLUE, fs=9, bold=False, ec="#333333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, text="", style="->", ls="-", fs=8, off=(0, 0.08)):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, lw=1.2, ls=ls, color="#222"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text,
                ha="center", va="bottom", fontsize=fs)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = canvas(12, 7.2, "System Architecture (layered)")
    box(ax, 0.3, 6.0, 11.4, 0.7, "Presentation layer:  main.py  (argparse CLI: scan | batch | demo | evaluate)",
        GREY, 10, True)
    box(ax, 0.3, 4.9, 11.4, 0.7, "Orchestration layer:  pipeline.py  ScanPipeline  ->  ScanResult", ORANGE, 10, True)
    ax.text(0.35, 4.55, "Processing modules (the three functional modules)", fontsize=9, style="italic")
    box(ax, 0.3, 2.9, 3.6, 1.5, "1. DETECTION\npreprocess.py\nedges.py (Canny, Hough)\ndetect.py", BLUE, 9, True)
    box(ax, 4.2, 2.9, 3.6, 1.5, "2. RECTIFICATION\ngeometry.py\nDLT homography,\nperspective warp", GREEN, 9, True)
    box(ax, 8.1, 2.9, 3.6, 1.5, "3. ENHANCEMENT\nenhance.py\nillumination fix, CLAHE,\nsharpen, adaptive threshold", ORANGE, 9, True)
    box(ax, 0.3, 1.5, 5.6, 0.9, "metrics.py : sharpness, contrast, entropy, corner error", GREY, 9)
    box(ax, 6.1, 1.5, 5.6, 0.9, "evaluate.py + synthetic.py : ground-truth evaluation", GREY, 9)
    box(ax, 0.3, 0.2, 3.6, 0.9, "io_utils.py : validation, load/save", GREY, 9)
    box(ax, 4.2, 0.2, 3.6, 0.9, "config.py : ScanConfig (validated)", GREY, 9)
    box(ax, 8.1, 0.2, 3.6, 0.9, "logger.py : console + file logs", GREY, 9)
    for x in (2.1, 6.0, 9.9):
        arrow(ax, (x, 4.9), (x, 4.4)); 
    arrow(ax, (6, 6.0), (6, 5.6))
    arrow(ax, (3.9, 3.65), (4.2, 3.65)); arrow(ax, (7.8, 3.65), (8.1, 3.65))
    save(fig, "architecture.png")


def workflow():
    fig, ax = canvas(11, 12.2, "Process / Workflow Diagram")
    cx, bw = 1.4, 4.8                      # main column left edge / width
    mid = cx + bw / 2
    rx, rw = 7.3, 3.3                      # side column (fallback branch)
    main = [(11.2, "Input image (file path)", GREY),
            (10.25, "Validate: exists, format, decodable, size", BLUE),
            (9.3, "Downscale to <=1000 px, grayscale, Gaussian blur", BLUE),
            (8.35, "Canny edges (Otsu thresholds)", BLUE),
            (7.1, "Contours: largest convex 4-gon found?", ORANGE),
            (5.9, "Order corners TL,TR,BR,BL; rescale to original", GREEN),
            (4.8, "Estimate homography H (normalised DLT + SVD)", GREEN),
            (3.7, "Warp to fronto-parallel page (optional A4/Letter ratio)", GREEN),
            (2.6, "Enhance: illumination fix + CLAHE / sharpen / threshold", ORANGE),
            (1.5, "Quality metrics (before vs after)", BLUE),
            (0.4, "Save scanned image (+ debug images) and log", GREY)]
    for y, t, c in main:
        box(ax, cx, y, bw, 0.62, t, c, 8.5, bold=(y == 7.1))
    for (y1, _, _), (y2, _, _) in zip(main, main[1:]):
        arrow(ax, (mid, y1), (mid, y2 + 0.62), "yes" if y1 == 7.1 else "", off=(0.3, -0.05))
    box(ax, rx, 7.1, rw, 0.62, "Hough-line fallback\n(outer lines + edge-support gate)", RED, 8)
    box(ax, rx, 5.9, rw, 0.62, "Quad found?", ORANGE, 9, True)
    box(ax, rx, 4.8, rw, 0.62, "Full-frame quad (log warning)", RED, 8)
    arrow(ax, (cx + bw, 7.41), (rx, 7.41), "no")
    arrow(ax, (rx + rw / 2, 7.1), (rx + rw / 2, 6.52))
    arrow(ax, (rx, 6.21), (cx + bw, 6.21), "yes")
    arrow(ax, (rx + rw / 2, 5.9), (rx + rw / 2, 5.42), "no", off=(0.25, 0))
    arrow(ax, (rx, 5.11), (cx + bw, 5.11))
    save(fig, "workflow.png")


def use_case():
    fig, ax = canvas(11, 6.6, "Use Case Diagram")
    ax.add_patch(FancyBboxPatch((3.2, 0.3), 7.5, 5.8, boxstyle="round,pad=0.02", fc="white", ec="#333", lw=1.5))
    ax.text(6.95, 5.8, "Document Scanner & Enhancer (CLI)", ha="center", fontsize=10, fontweight="bold")
    left = [(5.4, 5.0, "Scan single image"), (5.4, 4.15, "Batch-scan folder"),
            (5.4, 3.3, "Run synthetic demo"), (5.4, 2.45, "Evaluate accuracy"),
            (5.4, 1.6, "Choose mode / paper size"), (5.4, 0.8, "View logs & debug images")]
    for x, y, t in left:
        ax.add_patch(Ellipse((x, y), 3.0, 0.68, fc=BLUE, ec="#333"))
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5)
        arrow(ax, (1.3, 3.0), (x - 1.5, y), style="-", off=(0, 0))
    ax.add_patch(Ellipse((9.2, 3.65), 2.6, 0.9, fc=GREEN, ec="#333"))
    ax.text(9.2, 3.65, "Run scan\npipeline", ha="center", va="center", fontsize=8.5)
    for _, y, _ in left[:4]:
        arrow(ax, (6.9, y), (8.0, 3.65 + (y - 3.65) * 0.25), ls="--")
    ax.text(7.55, 4.9, "<<include>>", fontsize=7.5, style="italic")
    ax.add_patch(Ellipse((1.3, 3.75), 0.4, 0.4, fc="white", ec="#222"))
    ax.plot([1.3, 1.3], [3.55, 2.9], color="#222"); ax.plot([0.95, 1.65], [3.35, 3.35], color="#222")
    ax.plot([1.3, 1.0], [2.9, 2.45], color="#222"); ax.plot([1.3, 1.6], [2.9, 2.45], color="#222")
    ax.text(1.3, 2.15, "Student / User", ha="center", fontsize=9, fontweight="bold")
    save(fig, "use_case.png")


def sequence():
    names = ["User", "main.py\n(CLI)", "ScanPipeline", "detect.py", "geometry.py", "enhance.py", "io_utils.py"]
    fig, ax = canvas(13.6, 8, "Sequence Diagram: scan one image")
    xs = [0.9 + i * 1.9 for i in range(len(names))]
    for x, n in zip(xs, names):
        box(ax, x - 0.8, 7.0, 1.6, 0.7, n, GREY, 8, True)
        ax.plot([x, x], [0.3, 7.0], ls="--", color="#888", lw=1)
    msgs = [(0, 1, "scan photo.jpg --mode bw", 6.6), (1, 6, "load_image(path)", 6.2),
            (6, 1, "BGR array (validated)", 5.9), (1, 2, "run(image)", 5.5),
            (2, 3, "detect_document(image)", 5.1), (3, 2, "Detection(quad, method)", 4.7),
            (2, 4, "rectify(image, quad)", 4.3), (4, 2, "(rectified, H)", 3.9),
            (2, 5, "enhance(rectified, config)", 3.5), (5, 2, "enhanced image", 3.1),
            (2, 1, "ScanResult (+ metrics)", 2.7), (1, 6, "save_image(...)", 2.3),
            (1, 0, "path + contrast before->after", 1.9)]
    for a, b, t, y in msgs:
        ls = "--" if xs[b] < xs[a] else "-"
        arrow(ax, (xs[a], y), (xs[b], y), t, ls=ls, fs=7.5, off=(0, 0.04))
    ax.text(0.2, 0.7, "Errors (missing/corrupt file, bad config) raise ImageLoadError / ValueError;\n"
            "the CLI catches them, prints 'error: ...' and exits with a non-zero code.", fontsize=8, style="italic")
    save(fig, "sequence.png")


def class_diagram():
    fig, ax = canvas(13, 5.9, "Class / Component Diagram")

    def cls(x, y, w, name, attrs, methods, fc):
        h = 0.35 * (len(attrs) + len(methods)) + 0.75
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec="#333", lw=1.2))
        ax.text(x + w / 2, y + h - 0.25, name, ha="center", fontweight="bold", fontsize=9.5)
        ax.plot([x, x + w], [y + h - 0.45] * 2, color="#333", lw=0.8)
        yy = y + h - 0.7
        for a in attrs:
            ax.text(x + 0.1, yy, a, fontsize=7.5, va="center"); yy -= 0.35
        ax.plot([x, x + w], [yy + 0.17] * 2, color="#333", lw=0.8)
        for m in methods:
            ax.text(x + 0.1, yy - 0.05, m, fontsize=7.5, va="center"); yy -= 0.35
        return (x + w / 2, y + h / 2, x, y, w, h)

    cfg = cls(0.2, 3.3, 3.6, "ScanConfig  (frozen dataclass)",
              ["max_process_dim, blur_kernel", "min_area_ratio, hough_threshold_ratio",
               "clahe_clip, clahe_tiles, mode, paper"], ["__post_init__()  # validation"], BLUE)
    pipe = cls(4.6, 3.6, 4.0, "ScanPipeline", ["config: ScanConfig", "log: Logger"],
               ["run(image) -> ScanResult"], ORANGE)
    res = cls(9.3, 3.3, 3.5, "ScanResult  (dataclass)",
              ["rectified, enhanced, quad", "method, homography", "metrics: dict, overlay, edges"], [], GREEN)
    det = cls(0.2, 0.3, 3.8, "detect.py  ->  Detection",
              ["quad, method, edges"], ["detect_document(img, cfg)", "_quad_from_contours()", "_quad_from_hough()"], BLUE)
    geo = cls(4.4, 0.3, 4.0, "geometry.py",
              [], ["order_points()", "compute_homography_dlt()", "apply_homography()", "rectify(img, quad)"], GREEN)
    enh = cls(8.8, 0.3, 4.0, "enhance.py / metrics.py",
              [], ["correct_illumination(), apply_clahe()", "sharpen(), binarize(), enhance()", "quality_report(), corner_error()"], ORANGE)
    arrow(ax, (3.8, 4.6), (4.6, 4.6), "uses", off=(0, 0.05))
    arrow(ax, (8.6, 4.6), (9.3, 4.6), "returns", off=(0, 0.05))
    arrow(ax, (5.6, 3.6), (2.6, 2.55), "calls", ls="--")
    arrow(ax, (6.6, 3.6), (6.4, 2.55), "calls", ls="--")
    arrow(ax, (7.6, 3.6), (10.0, 2.55), "calls", ls="--")
    save(fig, "class_component.png")


if __name__ == "__main__":
    for fn in (architecture, workflow, use_case, sequence, class_diagram):
        fn()
    print("diagrams written to", OUT)
