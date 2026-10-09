#!/usr/bin/env python3
"""Command-line interface for the Document Scanner & Enhancer.

Examples
  python main.py demo     -o outputs/demo
  python main.py scan     photo.jpg -o outputs/scan --mode bw
  python main.py batch    photos/   -o outputs/batch
  python main.py evaluate -n 20 -o outputs/eval
"""
import argparse
import os
import sys

import numpy as np

from docscan import __version__
from docscan.config import PAPER_ASPECT, VALID_MODES, ScanConfig
from docscan.evaluate import run_evaluation
from docscan.io_utils import ImageLoadError, list_images, load_image, save_image
from docscan.logger import setup_logger
from docscan.pipeline import ScanPipeline
from docscan.synthetic import generate_document, make_scene


def _scan_one(pipeline, path, out_dir, debug):
    name = os.path.splitext(os.path.basename(path))[0]
    result = pipeline.run(load_image(path))
    out_path = save_image(os.path.join(out_dir, f"{name}_scanned.png"), result.enhanced)
    if debug:
        save_image(os.path.join(out_dir, f"{name}_edges.png"), result.edges)
        save_image(os.path.join(out_dir, f"{name}_detected.png"), result.overlay)
        save_image(os.path.join(out_dir, f"{name}_rectified.png"), result.rectified)
    b, a = result.metrics["before"], result.metrics["after"]
    print(f"{os.path.basename(path)} -> {out_path} | method={result.method} | "
          f"contrast {b['contrast']:.1f}->{a['contrast']:.1f}")
    return result


def cmd_scan(args, cfg):
    _scan_one(ScanPipeline(cfg), args.input, args.output, args.debug)
    return 0


def cmd_batch(args, cfg):
    paths = list_images(args.folder)
    if not paths:
        print(f"error: no supported images in {args.folder}", file=sys.stderr)
        return 1
    pipeline, failures = ScanPipeline(cfg), 0
    for path in paths:                       # one bad file must not stop the batch
        try:
            _scan_one(pipeline, path, args.output, args.debug)
        except (ImageLoadError, ValueError, IOError) as exc:
            failures += 1
            print(f"skipped {os.path.basename(path)}: {exc}", file=sys.stderr)
    print(f"done: {len(paths) - failures}/{len(paths)} scanned")
    return 0 if failures == 0 else 2


def cmd_demo(args, cfg):
    rng = np.random.default_rng(args.seed)
    scene, _ = make_scene(generate_document(rng), rng)
    save_image(os.path.join(args.output, "demo_input.png"), scene)
    _scan_one(ScanPipeline(cfg), os.path.join(args.output, "demo_input.png"),
              args.output, debug=True)
    return 0


def cmd_evaluate(args, cfg):
    s = run_evaluation(args.n, args.output, cfg, seed0=args.seed)
    print(f"scenes: {s['scenes']}  success rate (<=10px): {s['success_rate']:.0%}")
    print(f"mean corner error: {s['mean_corner_error_px']:.2f}px  "
          f"max: {s['max_corner_error_px']:.2f}px  "
          f"contour-method share: {s['contour_method_share']:.0%}")
    print(f"ablation - Hough-only fallback success rate (<=10px): {s['hough_only_success_rate']:.0%}")
    print(f"per-scene results: {s['csv']}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="docscan", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"docscan {__version__}")
    p.add_argument("--log-file", default="logs/docscan.log", help="log file path")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("-o", "--output", default="outputs", help="output folder")
        sp.add_argument("--mode", choices=VALID_MODES, default="color", help="enhancement mode")
        sp.add_argument("--paper", choices=list(PAPER_ASPECT), default="auto",
                        help="paper-size aspect prior (default: estimate from the photo)")
        sp.add_argument("--debug", action="store_true", help="also save edges/overlay/rectified")

    s = sub.add_parser("scan", help="scan a single image"); s.add_argument("input"); common(s)
    s.set_defaults(func=cmd_scan)
    b = sub.add_parser("batch", help="scan every image in a folder"); b.add_argument("folder"); common(b)
    b.set_defaults(func=cmd_batch)
    d = sub.add_parser("demo", help="generate a synthetic photo and scan it"); common(d)
    d.add_argument("--seed", type=int, default=7); d.set_defaults(func=cmd_demo)
    e = sub.add_parser("evaluate", help="measure accuracy on synthetic scenes"); common(e)
    e.add_argument("-n", type=int, default=20, help="number of scenes")
    e.add_argument("--seed", type=int, default=0); e.set_defaults(func=cmd_evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logger(args.log_file, args.verbose)
    try:
        return args.func(args, ScanConfig(mode=args.mode, paper=args.paper))
    except (ImageLoadError, ValueError, IOError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
