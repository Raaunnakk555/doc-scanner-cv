"""Central, validated configuration for the scanning pipeline."""
from dataclasses import dataclass

VALID_MODES = ("color", "gray", "bw")
# width / height ratios for known paper sizes (None = estimate from edge lengths)
PAPER_ASPECT = {"auto": None, "a4": 210 / 297, "letter": 8.5 / 11}


@dataclass(frozen=True)
class ScanConfig:
    """All tunable parameters live here so no magic numbers hide in the code."""

    max_process_dim: int = 1000        # detection runs on a downscaled copy
    blur_kernel: int = 5               # Gaussian blur kernel (odd)
    min_area_ratio: float = 0.10       # document must cover >= 10% of the frame
    approx_epsilon_ratio: float = 0.02  # polygon approximation tolerance
    hough_threshold_ratio: float = 0.15  # Hough votes needed, relative to image size
    clahe_clip: float = 2.0            # CLAHE contrast limit
    clahe_tiles: int = 8               # CLAHE tile grid size
    illum_kernel_ratio: float = 0.08   # background-estimation kernel / image size
    adaptive_block: int = 31           # adaptive threshold neighbourhood (odd)
    adaptive_c: int = 12               # adaptive threshold constant
    mode: str = "color"                # one of VALID_MODES
    paper: str = "auto"                # one of PAPER_ASPECT keys

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        if self.paper not in PAPER_ASPECT:
            raise ValueError(f"paper must be one of {tuple(PAPER_ASPECT)}, got {self.paper!r}")
        if self.blur_kernel % 2 == 0 or self.blur_kernel < 1:
            raise ValueError("blur_kernel must be a positive odd integer")
        if self.adaptive_block % 2 == 0 or self.adaptive_block < 3:
            raise ValueError("adaptive_block must be an odd integer >= 3")
        if not 0.0 < self.min_area_ratio < 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1")
        if self.max_process_dim < 100:
            raise ValueError("max_process_dim must be >= 100")
