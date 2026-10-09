"""Synthetic test data: a fake document photographed at a random angle.

Because we create the perspective warp ourselves, the exact ground-truth
corners are known, which lets us measure detection error numerically.
"""
import cv2
import numpy as np

WORDS = ("vision image pixel camera lens depth stereo edge corner scale filter "
         "gradient kernel feature motion pattern epipolar homography texture").split()


def generate_document(rng, width=700, height=900):
    """White page with a title, text-like lines and a table."""
    doc = np.full((height, width, 3), 247, np.uint8)
    cv2.putText(doc, "Computer Vision Notes", (50, 90), cv2.FONT_HERSHEY_DUPLEX,
                1.3, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(doc, (50, 110), (width - 50, 110), (60, 60, 60), 2)
    y = 160
    while y < height - 260:
        line = " ".join(rng.choice(WORDS, size=int(rng.integers(5, 9))))
        cv2.putText(doc, line[:48], (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (40, 40, 40), 1, cv2.LINE_AA)
        y += 34
    top = height - 230
    for r in range(4):
        for c in range(3):
            x0, y0 = 50 + c * 200, top + r * 45
            cv2.rectangle(doc, (x0, y0), (x0 + 200, y0 + 45), (70, 70, 70), 1)
            cv2.putText(doc, f"{WORDS[(r * 3 + c) % len(WORDS)]}", (x0 + 12, y0 + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return doc


def make_scene(doc, rng, canvas=(1200, 1000)):
    """Place the document on a textured table in perspective.

    Returns (scene_bgr, ground_truth_corners) with corners ordered TL,TR,BR,BL.
    """
    cw, ch = canvas
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    margin_x, margin_y, jitter = 0.18 * cw, 0.15 * ch, 0.06 * cw
    base = np.float32([[margin_x, margin_y], [cw - margin_x, margin_y],
                       [cw - margin_x, ch - margin_y], [margin_x, ch - margin_y]])
    dst = base + rng.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)

    # smooth, darker textured "table" so the paper edge stays strong
    colour = rng.integers(40, 110, size=3).astype(np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (ch, cw)).astype(np.float32), (0, 0), 25) * 400
    background = np.clip(colour[None, None, :] + noise[..., None], 0, 255).astype(np.uint8)

    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, H, (cw, ch), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (cw, ch))
    scene = np.where(mask[..., None] > 0, warped, background).astype(np.float32)

    # uneven lighting (soft gradient) + sensor noise
    angle = rng.uniform(0, 2 * np.pi)
    xx, yy = np.meshgrid(np.linspace(-1, 1, cw), np.linspace(-1, 1, ch))
    shade = 0.85 + 0.25 * (np.cos(angle) * xx + np.sin(angle) * yy) / 2
    scene = scene * shade[..., None] + rng.normal(0, 4, scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8), dst
