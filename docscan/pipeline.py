"""End-to-end scanning pipeline: detect -> rectify -> enhance -> measure."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PAPER_ASPECT, ScanConfig
from .detect import detect_document
from .enhance import enhance
from .geometry import rectify
from .io_utils import validate_image_array
from .logger import get_logger
from .metrics import quality_report


@dataclass
class ScanResult:
    rectified: np.ndarray
    enhanced: np.ndarray
    quad: np.ndarray
    method: str
    homography: np.ndarray
    metrics: dict = field(default_factory=dict)
    overlay: np.ndarray = None      # input with detected quad drawn on it
    edges: np.ndarray = None


class ScanPipeline:
    def __init__(self, config=None):
        self.config = config or ScanConfig()
        self.log = get_logger()

    def run(self, image):
        validate_image_array(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        det = detect_document(image, self.config)
        self.log.info("detection method: %s", det.method)

        rectified, H = rectify(image, det.quad, PAPER_ASPECT[self.config.paper])
        self.log.info("rectified to %dx%d", rectified.shape[1], rectified.shape[0])

        enhanced = enhance(rectified, self.config)
        metrics = {"before": quality_report(rectified), "after": quality_report(enhanced)}

        overlay = image.copy()
        cv2.polylines(overlay, [det.quad.astype(np.int32)], True, (0, 255, 0),
                      max(2, image.shape[1] // 300))
        return ScanResult(rectified, enhanced, det.quad, det.method, H,
                          metrics, overlay, det.edges)
