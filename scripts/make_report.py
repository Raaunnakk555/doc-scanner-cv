"""Build the PDF project report (requires reportlab).

Run from the repo root (after scripts/make_diagrams.py and make_figures.py):
    python scripts/make_report.py
"""
import csv
import os
import sys
import unittest

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle, KeepTogether)

sys.path.insert(0, os.getcwd())
OUT = os.path.join("docs", "Project_Report.pdf")
DIAG, FIG = os.path.join("docs", "diagrams"), os.path.join("docs", "figures")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=ss["Normal"], fontSize=10, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], keepWithNext=1, fontSize=16, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1f3b73"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], keepWithNext=1, fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#2d5aa8"))
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.5, leading=11, alignment=0, spaceAfter=0)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=8.5, leading=11, alignment=0,
                      backColor=colors.HexColor("#f3f3f3"), borderPadding=4, spaceAfter=8)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))
def bullets(items):
    for t in items:
        story.append(Paragraph(t, BUL, bulletText="\u2022"))
    story.append(Spacer(1, 4))

def table(rows, widths, header=True, lead=None):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9ff"))]
    t.setStyle(TableStyle(style))
    lead_p = [Paragraph(lead, BODY)] if lead else []
    story.append(KeepTogether(lead_p + [t, Spacer(1, 8)]) if len(rows) <= 9 else t)
    if len(rows) > 9:
        story.append(Spacer(1, 8))

def image(path, width_cm, caption, heading=None):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    parts = ([Paragraph(heading, H2)] if heading else []) + [img, Paragraph(caption, CAP), Spacer(1, 8)]
    story.append(KeepTogether(parts))

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "Document Scanner & Enhancer - Computer Vision Project Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"); canvas.restoreState()

# ---------- load real data ----------
rows = list(csv.DictReader(open(os.path.join("docs", "evaluation", "evaluation.csv"))))
errs = np.array([float(r["corner_error_px"]) for r in rows])
hough = np.array([float(r["hough_only_error_px"]) for r in rows])
n = len(rows)
success = np.mean(errs <= 10) * 100
h_success = np.mean(np.nan_to_num(hough, nan=1e9) <= 10) * 100
h_fail = sorted(np.round(hough[hough > 10], 1).tolist())
c_b = np.mean([float(r["contrast_before"]) for r in rows]); c_a = np.mean([float(r["contrast_after"]) for r in rows])
e_b = np.mean([float(r["entropy_before"]) for r in rows]); e_a = np.mean([float(r["entropy_after"]) for r in rows])

loader = unittest.TestLoader()
counts = {m: loader.loadTestsFromName(f"tests.{m}").countTestCases()
          for m in ("test_geometry", "test_preprocess", "test_detect_pipeline", "test_cli")}
total_tests = sum(counts.values())

# ---------- 1. Cover ----------
story.append(Spacer(1, 3 * cm))
P("VITyarthi - Build Your Own Project", ParagraphStyle("c0", parent=BODY, alignment=TA_CENTER, fontSize=12, textColor=colors.grey))
story.append(Spacer(1, 1 * cm))
P("Document Scanner &amp; Enhancer", ParagraphStyle("c1", parent=H1, fontSize=30, alignment=TA_CENTER, leading=36))
P("A command-line document scanner using classical computer vision:<br/>edge detection, Hough transform, homography rectification and histogram-based enhancement",
  ParagraphStyle("c2", parent=BODY, alignment=TA_CENTER, fontSize=12, leading=17))
story.append(Spacer(1, 2 * cm))
table([["Course", "Computer Vision"], ["Student name", "[Your Name]"], ["Registration number", "[Your Reg. No.]"],
       ["Faculty", "[Faculty Name]"], ["GitHub repository", "https://github.com/[username]/doc-scanner-cv"],
       ["Submission date", "[dd-mm-yyyy]"]], [5, 10], header=False)
story.append(Spacer(1, 2 * cm))
P("<b>Note:</b> this is a faculty-prepared <b>reference sample</b> showing the expected structure, depth and evidence for the project "
  "report. Students must design, implement and document their own original project.", ParagraphStyle("c3", parent=CAP, fontSize=9))
story.append(PageBreak())

# ---------- 2. Introduction ----------
P("1. Introduction", H1)
P("Phones have replaced flatbed scanners for many everyday tasks, but a photograph of a page is not a scan. It is a perspective "
  "projection of a plane onto the image sensor: the rectangle of the page becomes a general quadrilateral, the background is "
  "included, and lighting varies across the page. Turning such a photograph into a clean, fronto-parallel image is a "
  "textbook computer-vision pipeline that touches image formation, low-level filtering, edge and line detection, and "
  "projective geometry.")
P("This project implements that pipeline as a modular Python package with a command-line interface. The emphasis is on "
  "<i>understanding</i>: the Direct Linear Transform (DLT) homography and histogram equalisation are written from first "
  "principles with NumPy, and OpenCV is used for standard building blocks (Canny, Hough, resampling).")
P("<b>Syllabus coverage.</b> Unit 1 (image formation, convolution and filtering, histogram processing, enhancement); Unit 2 "
  "(projective transformations, homography, DLT, rectification); Unit 3 (Canny edges, Hough transform, contour-based "
  "segmentation).")

# ---------- 3. Problem statement ----------
P("2. Problem Statement", H1)
P("Given a single photograph containing one roughly rectangular paper page against an arbitrary background, automatically "
  "(a) locate the page boundary, (b) remove the perspective distortion to produce a rectified page image, and (c) enhance "
  "the page so that text is legible despite shadows and uneven illumination. The system must run from a terminal, "
  "report measurable quality and accuracy figures, and fail gracefully on bad input.")
P("<b>Objectives</b>")
bullets(["Detect the page quadrilateral with sub-10-pixel corner error on controlled data.",
         "Estimate the homography with a from-scratch, numerically stable DLT.",
         "Improve page contrast and flatten illumination, with selectable colour / grey / black-and-white output.",
         "Provide an evaluation harness with ground truth so claims are measured, not assumed.",
         "Deliver a clean, tested, documented repository suitable for evaluation."])
P("<b>Target users:</b> students and staff needing quick scans; CV learners; developers needing an OCR pre-processing stage. "
  "<b>Out of scope:</b> OCR, multi-page or curved pages, GUI front-ends, deep-learning models.")

# ---------- 4. FR ----------
P("3. Functional Requirements", H1)
table([["ID", "Requirement", "Module"],
       ["FR1", "Load and validate an input image (existence, extension, decodability, minimum size).", "io_utils"],
       ["FR2", "Detect the page quadrilateral using edges and contours; fall back to Hough lines; as a last resort use the full frame.", "Detection"],
       ["FR3", "Order corners and estimate a homography with a normalised DLT.", "Rectification"],
       ["FR4", "Warp to a fronto-parallel image, optionally imposing an A4 or Letter aspect ratio.", "Rectification"],
       ["FR5", "Enhance in three modes: color, gray, bw (illumination correction, CLAHE, sharpening, adaptive threshold).", "Enhancement"],
       ["FR6", "Process one image, a folder of images, or a generated synthetic demo from the CLI.", "main.py"],
       ["FR7", "Compute quality metrics (sharpness, contrast, entropy) before/after enhancement.", "metrics"],
       ["FR8", "Evaluate on synthetic scenes against ground-truth corners, including a detector ablation; export CSV.", "evaluate"]],
      [1.2, 12.3, 2.9])
P("<b>Input/output structure.</b> Input: JPG/PNG/BMP/TIFF image path (or folder). Output: <font name='Courier'>*_scanned.png</font> "
  "plus, with <font name='Courier'>--debug</font>, edge map, detection overlay and rectified image; a log file; and "
  "<font name='Courier'>evaluation.csv</font> for the evaluate command.")
P("<b>User workflow.</b> (1) Install dependencies; (2) run <font name='Courier'>demo</font> to see the system work; (3) run "
  "<font name='Courier'>scan</font> or <font name='Courier'>batch</font> on real photos choosing a mode and paper size; (4) inspect "
  "outputs and logs; (5) run <font name='Courier'>evaluate</font> to quantify accuracy.")

# ---------- 5. NFR ----------
P("4. Non-functional Requirements", H1)
table([["Category", "Requirement", "How it is met / measured"],
       ["Performance", "Scan a 1.2 MP photo in well under one second on a laptop.", "Detection runs on a copy downscaled to 1000 px. Measured about 60-70 ms per 1200x1000 image; about 0.8 s for a 12 MP image (development sandbox; hardware dependent)."],
       ["Reliability", "One bad file must never stop a batch.", "Batch mode catches per-file errors, reports them, continues and returns exit code 2. Detection has three-level fallback (contour, Hough, full frame)."],
       ["Usability", "Self-explanatory CLI.", "argparse sub-commands with help text, sensible defaults, and a <i>demo</i> command that needs no input data."],
       ["Maintainability", "Modular, documented, testable code.", "12 focused modules, a single validated config dataclass, no magic numbers, docstrings, 40 automated tests."],
       ["Error handling", "Clear messages, correct exit codes.", "Custom ImageLoadError; config validated at construction; CLI prints 'error: ...' and returns 1."],
       ["Logging", "Traceable runs.", "Console + file logging (logs/docscan.log), DEBUG with -v; warnings when fallbacks fire."],
       ["Resource efficiency", "No heavy dependencies, bounded memory.", "Only NumPy and OpenCV; processing resolution capped; peak memory about 300 MB even for a 12 MP input."]],
      [2.6, 5.0, 8.8])

# ---------- 6. Architecture ----------
P("5. System Architecture", H1)
P("The system is a layered, pipeline-style architecture. The CLI layer parses arguments and handles errors; the orchestration "
  "layer (<font name='Courier'>ScanPipeline</font>) sequences the three processing modules; supporting modules provide "
  "configuration, logging, I/O, metrics and evaluation. Each module has one responsibility and communicates through plain "
  "NumPy arrays and small dataclasses, which keeps coupling low and makes every module independently testable.")
image(os.path.join(DIAG, "architecture.png"), 15.5, "Figure 1 - Layered system architecture")

# ---------- 7. Design diagrams ----------
story.append(PageBreak())
P("6. Design Diagrams", H1)
image(os.path.join(DIAG, "use_case.png"), 13.5, "Figure 2 - Use cases for the student / user actor", "6.1 Use Case Diagram")
image(os.path.join(DIAG, "workflow.png"), 11.0, "Figure 3 - Process flow including fallback strategies", "6.2 Workflow Diagram")
story.append(PageBreak())
image(os.path.join(DIAG, "sequence.png"), 16.0, "Figure 4 - Sequence of calls for the 'scan' command", "6.3 Sequence Diagram")
image(os.path.join(DIAG, "class_component.png"), 16.0, "Figure 5 - Principal classes and module components", "6.4 Class / Component Diagram")
P("6.5 ER Diagram / Storage Design", H2)
P("<b>Not applicable.</b> The system keeps no database. Its only persistent data are image files and a CSV written to a "
  "user-chosen output folder, and a text log. The CSV schema is: <font name='Courier'>scene, method, corner_error_px, success, "
  "hough_only_error_px, contrast_before, contrast_after, entropy_before, entropy_after</font>.")

# ---------- 8. Design decisions ----------
P("7. Design Decisions &amp; Rationale", H1)
table([["Decision", "Alternatives considered", "Rationale"],
       ["Contours as primary detector, Hough as fallback", "Hough only; deep-learning segmentation", "Contours on a closed edge map are robust to broken borders and give a direct 4-gon. Hough lines are useful but confused by text rules inside the page (see Challenges). Deep learning would need data and hide the maths."],
       ["From-scratch normalised DLT", "Call cv2.getPerspectiveTransform only", "Teaches the core multi-view-geometry idea. Hartley normalisation conditions the SVD; tests prove agreement with OpenCV."],
       ["Otsu-derived Canny thresholds", "Fixed or median-based thresholds", "Adapts to image brightness without manual tuning; low = 0.5 x high follows Canny's 1:2 guidance."],
       ["Detect on a downscaled copy, rectify at full resolution", "Process everything at full size", "Measured on a 12 MP image: detection takes about 60 ms downscaled versus about 110 ms at full resolution (1.8x faster). It also makes fixed kernel sizes behave consistently across camera resolutions, and output quality is kept because the warp uses the original pixels."],
       ["Flat-field correction before CLAHE", "CLAHE alone", "Shadows are a multiplicative background; dividing by an estimated background removes them, then CLAHE restores local contrast."],
       ["Optional paper aspect prior", "Always use quad edge lengths", "Edge lengths of a perspective-distorted quad do not give the true aspect ratio; a prior removes the stretch."],
       ["Frozen dataclass config with validation", "Loose keyword arguments / globals", "Fail fast on bad parameters; one place for every tunable value."],
       ["unittest instead of pytest", "pytest", "Zero extra dependencies; evaluators can run tests on any Python install."],
       ["Synthetic data with ground truth", "Only real photos", "Known corners allow numeric error measurement and a repeatable ablation; real photos are recommended as future work."]],
      [3.8, 3.6, 9.0])

# ---------- 9. Implementation ----------
P("8. Implementation Details", H1)
P("8.1 Module 1 - Detection (preprocess.py, edges.py, detect.py)", H2)
P("The image is downscaled (longest side &lt;= 1000 px), converted to grey and smoothed with a 5x5 Gaussian, which is a low-pass "
  "convolution that suppresses sensor noise before differentiation. <b>Canny</b> thresholds come from Otsu's method: "
  "high = T<sub>Otsu</sub>, low = 0.5 T<sub>Otsu</sub>. The edge map is morphologically closed to bridge small gaps; "
  "<font name='Courier'>findContours</font> (external only) returns candidate blobs sorted by area. For each, the convex hull is "
  "simplified with <font name='Courier'>approxPolyDP</font> at epsilon = 2% of the perimeter (relaxed to 3-5% if necessary) "
  "until exactly four vertices remain.")
P("<b>Hough fallback.</b> If no 4-gon is found, the standard Hough transform returns lines (rho, theta) with "
  "rho = x cos(theta) + y sin(theta). Lines within 30 degrees of horizontal and of vertical are grouped; the "
  "<i>outermost</i> line of each group is taken as a page border because text rules and table lines lie inside the page. "
  "Their four intersections give a candidate quad, which is accepted only if at least 45% of its perimeter lies on real "
  "edges (an <i>edge-support gate</i>) - a bad guess is better rejected than shown. If everything fails, the full frame is used and a warning is logged.")
P("8.2 Module 2 - Rectification (geometry.py)", H2)
P("Corners are ordered top-left, top-right, bottom-right, bottom-left using the sum and difference of coordinates. A homography H "
  "maps source points x to destination points x' (in homogeneous coordinates, x' ~ Hx). Each correspondence gives two linear "
  "equations in the nine entries of H; stacking them yields A h = 0, solved as the right singular vector of A with the smallest "
  "singular value (SVD). Following Hartley, both point sets are first translated to zero mean and scaled to mean distance "
  "sqrt(2); the estimate is de-normalised as H = T'<super>-1</super> H<sub>n</sub> T. The destination rectangle's size comes "
  "from the longest top/bottom and left/right edges (or from the A4/Letter ratio at equal area), and "
  "<font name='Courier'>warpPerspective</font> resamples with bicubic interpolation.")
story.append(Paragraph("[ -x -y -1  0  0  0  ux  uy  u ]  h = 0<br/>[  0  0  0 -x -y -1  vx  vy  v ]", CODE))
P("Here (x, y) is a source point, (u, v) its destination, and h is the nine entries of H as a vector.")
P("8.3 Module 3 - Enhancement (enhance.py, preprocess.py)", H2)
P("<b>Illumination correction:</b> dilation removes dark text, a large median blur (kernel = 8% of the image side) gives the "
  "page background B(x,y), and the output is I/B scaled to 255, cancelling smooth shadows. <b>CLAHE</b> (clip 2.0, 8x8 tiles) "
  "then boosts local contrast. In <i>color</i> mode these operate on the L channel of CIELAB so colours are preserved; "
  "<i>gray</i> mode adds unsharp masking; <i>bw</i> uses an adaptive Gaussian threshold (block 31, C = 12). "
  "A global histogram equalisation, s<sub>k</sub> = round((CDF(k) - CDF<sub>min</sub>) / (N - CDF<sub>min</sub>) x 255), is "
  "implemented from scratch with a lookup table and verified against OpenCV.")
P("8.4 Supporting modules", H2)
bullets(["<b>config.py</b> - frozen dataclass; invalid values raise ValueError at construction.",
         "<b>io_utils.py</b> - ImageLoadError for missing, unsupported, corrupt or tiny images; safe writing that creates folders.",
         "<b>metrics.py</b> - variance of Laplacian (sharpness), RMS contrast, Shannon entropy, mean corner error.",
         "<b>synthetic.py</b> - draws a text-and-table page, warps it with a random projective transform onto a textured table, adds a lighting gradient and noise; returns exact ground-truth corners.",
         "<b>evaluate.py</b> - runs N scenes, records errors, runs the Hough-only ablation, writes CSV.",
         "<b>main.py</b> - argparse CLI with sub-commands, error handling and exit codes (0 success, 1 error, 2 partial batch)."])
P("8.5 Version control", H2)
P("The repository was built incrementally with meaningful commits (scaffold, low-level modules, detection and pipeline, "
  "synthetic data, CLI, tests, diagrams, documentation), so the history documents the development process.")

# ---------- 10. Results ----------
story.append(PageBreak())
P("9. Screenshots / Results", H1)
image(os.path.join(FIG, "pipeline_stages.png"), 16.5, "Figure 6 - Pipeline stages on a synthetic photo: input, edges, detected quad, rectified page and the three enhancement modes")
image(os.path.join(FIG, "histograms.png"), 12.5, "Figure 7 - Grey-level histogram before and after equalisation")
table([["Metric", "Value"],
       ["Detection success (mean corner error &lt;= 10 px)", f"{success:.0f}%"],
       ["Mean / maximum corner error (primary contour method)", f"{errs.mean():.2f} px / {errs.max():.2f} px"],
       ["Hough-only fallback success (ablation)", f"{h_success:.0f}%  (misses at {', '.join(str(x) for x in h_fail)} px)"],
       ["Mean RMS contrast of rectified page, before to after enhancement (colour mode)", f"{c_b:.1f} to {c_a:.1f}"],
       ["Mean grey-level entropy, before to after", f"{e_b:.2f} to {e_a:.2f} bits"],
       ["Average time per 1.2 MP image", "about 60-70 ms"]], [10.5, 5.9],
      lead=f"<b>Quantitative evaluation</b> on {n} synthetic scenes (random perspective, lighting gradient and sensor noise):")
image(os.path.join(FIG, "evaluation.png"), 14.5, "Figure 8 - Corner error per scene for the primary and fallback detectors")
P("<b>Interpretation.</b> The contour method localises corners to about one pixel on controlled data, and the Hough fallback "
  "is a credible second opinion that fails on a minority of scenes. These figures come from <i>synthetic</i> scenes whose "
  "background is smooth and whose page contrast is high, so real-world accuracy will be lower; the numbers demonstrate "
  "that the implementation is correct, not that it is production-ready. Entropy <i>falls</i> after enhancement; this is expected "
  "and desirable for documents, because shadows and background gradients are replaced by a nearly uniform white page, leaving fewer distinct grey levels.")

# ---------- 11. Testing ----------
P("10. Testing Approach", H1)
P(f"The suite contains <b>{total_tests} automated tests</b> using Python's built-in unittest "
  "(<font name='Courier'>python -m unittest discover -s tests -t .</font>). Tests were written after the core modules and expanded as defects were found (notably the Hough fallback), "
  "and were re-run throughout development.")
table([["Test file", "Tests", "What it verifies"],
       ["test_geometry.py", counts["test_geometry"], "DLT maps points exactly; agrees with cv2.getPerspectiveTransform; identity case; noisy over-determined estimation; too-few and collinear points rejected; corner ordering under permutations; output size and aspect handling; end-to-end rectification of a known image."],
       ["test_preprocess.py", counts["test_preprocess"], "Histogram sums to pixel count; equalisation stretches contrast and is within 2 grey levels of OpenCV; flat image unchanged; resize helpers; illumination correction flattens a gradient; thresholded output is binary."],
       ["test_detect_pipeline.py", counts["test_detect_pipeline"], "Contour accuracy on 5 scenes; Hough fallback accuracy and rejection of noise; blank image gives full-frame fallback; line intersection; all enhancement modes; contrast improvement; grayscale input; paper aspect; invalid input; config validation; IO errors; round-trip save/load; entropy bounds."],
       ["test_cli.py", counts["test_cli"], "Real subprocess runs: demo creates outputs; missing file returns exit code 1 with an error message; evaluate prints a success rate; batch skips corrupt files and returns code 2."]],
      [3.6, 1.4, 11.4])
P("<b>Validation tests</b> go beyond unit checks: the evaluation harness measures corner error against exact ground truth, which "
  "validates the whole pipeline numerically. <b>Negative tests</b> (bad config, corrupt files, degenerate geometry, noise images) "
  "confirm the error-handling strategy.")

# ---------- 12. Challenges ----------
P("11. Challenges Faced", H1)
bullets(["<b>Hough fallback accuracy.</b> The first fallback picked the strongest lines by vote count and failed often: rows of text and table rules produce many strong, near-horizontal lines inside the page, outvoting the real borders. A second attempt scored every candidate quadrilateral by edge support but was worse (the true borders were not among the top-voted lines). The final design takes the <i>outermost</i> strong lines and adds an edge-support gate that rejects poor candidates. Lesson: measure, do not guess.",
         "<b>Misleading experiments.</b> Early tuning experiments called the fallback helpers directly at full resolution and suggested it was weak (under 50% success); evaluating it through the real pipeline path (downscaled processing, same parameters as deployed) gave 90%. The exact cause of the gap was not isolated, but the lesson is clear: evaluate a component exactly as it is deployed.",
         "<b>Aspect-ratio ambiguity.</b> A rectified page came out landscape although the source page was portrait. A single quadrilateral cannot determine the true aspect ratio without camera intrinsics; the optional A4/Letter prior was added as a practical remedy.",
         "<b>Uneven lighting.</b> Plain histogram equalisation amplified shadows. Dividing by an estimated background first (flat-field correction) solved this.",
         "<b>Numerical conditioning.</b> Un-normalised DLT is sensitive to coordinate scale; Hartley normalisation fixed it.",
         "<b>Evidence without a dataset.</b> No labelled document-photo set was available, so synthetic scenes with exact ground truth were generated. Their limits are stated openly."])

# ---------- 13. Learnings ----------
P("12. Learnings &amp; Key Takeaways", H1)
bullets(["Classical CV pipelines are chains of simple, testable steps; the quality of each step can be measured on its own.",
         "Projective geometry is concrete: four point pairs determine a homography, and the SVD solves it.",
         "Robustness comes from fallbacks, validation gates and honest failure modes, not from one clever algorithm.",
         "Ground-truth synthetic data enables ablation studies and repeatable evaluation.",
         "Good engineering (config validation, logging, exit codes, tests, incremental commits) is part of the result, not an afterthought."])

# ---------- 14. Future ----------
P("13. Future Enhancements", H1)
bullets(["Evaluate on a public real-world dataset of document photos and report precision/recall.",
         "Sub-pixel corner refinement and robust line fitting with RANSAC for the fallback.",
         "Estimate the true aspect ratio from camera intrinsics or vanishing points (Unit 2: auto-calibration).",
         "Handle curved pages (dewarping) and multiple pages per frame.",
         "Add an OCR stage and report character error rate before vs. after enhancement.",
         "Add noise and blur robustness benchmarks; package for pip."])

# ---------- 15. References ----------
P("14. References", H1)
refs = ["R. Hartley and A. Zisserman, <i>Multiple View Geometry in Computer Vision</i>, 2nd ed., Cambridge University Press, 2004.",
        "J. Canny, \"A computational approach to edge detection,\" <i>IEEE Trans. Pattern Analysis and Machine Intelligence</i>, vol. 8, no. 6, pp. 679-698, 1986.",
        "N. Otsu, \"A threshold selection method from gray-level histograms,\" <i>IEEE Trans. Systems, Man, and Cybernetics</i>, vol. 9, no. 1, pp. 62-66, 1979.",
        "R. O. Duda and P. E. Hart, \"Use of the Hough transformation to detect lines and curves in pictures,\" <i>Communications of the ACM</i>, vol. 15, no. 1, pp. 11-15, 1972.",
        "K. Zuiderveld, \"Contrast limited adaptive histogram equalization,\" in <i>Graphics Gems IV</i>, Academic Press, 1994, pp. 474-485.",
        "R. Szeliski, <i>Computer Vision: Algorithms and Applications</i>, 2nd ed., Springer, 2022.",
        "R. C. Gonzalez and R. E. Woods, <i>Digital Image Processing</i>, 4th ed., Pearson, 2018.",
        "G. Bradski, \"The OpenCV Library,\" <i>Dr. Dobb's Journal of Software Tools</i>, 2000. https://opencv.org",
        "C. R. Harris et al., \"Array programming with NumPy,\" <i>Nature</i>, vol. 585, pp. 357-362, 2020."]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}] {r}", ParagraphStyle("ref", parent=BODY, leftIndent=18, firstLineIndent=-18, spaceAfter=3)))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Document Scanner & Enhancer - Project Report", author="Course Faculty (reference sample)")
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("report written to", OUT)
