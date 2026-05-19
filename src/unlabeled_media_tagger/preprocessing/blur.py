"""Blur / quality scoring for face crops (v1.1 quality gate).

`laplacian_variance(bgr)` — variance of the Laplacian on the grayscale crop.
Higher = sharper. Classic, cheap focus metric.

`classify_quality(score, w, h, ...)` — encodes the v1.1 filter policy:
size first (defends against halftone-backdrop false-positives that pure blur
scoring cannot see), then blur score. Returns one of "ok", "filtered_small_crop",
"filtered_blurry".

Defaults (`DEFAULT_BLUR_THRESHOLD`, `DEFAULT_MIN_CROP_DIM`) were tuned against
real Converge debate photography on 2026-05-18; see dev-log/ for the run-by-run
rationale. They are reasonable starting points but not universal.

Known limitations:
- Score scales with crop texture, not size. Halftone-printed faces in
  background collages score high despite being out of focus — this is what
  the size filter exists to catch.
- Profile shots with glasses score lower than frontal shots of the same
  person (less visible facial-feature edges). Don't push the blur threshold
  so high that you over-filter profiles.
- Motion blur in dim light can produce moderate scores. Re-tune the
  threshold if you hit a dataset where this matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_BLUR_THRESHOLD = 45.0
DEFAULT_MIN_CROP_DIM = 100


def laplacian_variance(bgr: np.ndarray) -> float:
    """Variance of the Laplacian of `bgr` after grayscale conversion.

    Returns 0.0 for empty / None arrays so callers don't have to special-case
    those upstream. For real crops, returned value is strictly positive.
    """
    if bgr is None or bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_quality(
    score: float,
    width: int,
    height: int,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    min_crop_dim: int = DEFAULT_MIN_CROP_DIM,
) -> str:
    """Decide a v1.1 status for a detected face crop.

    Returns one of:
        "ok"                    crop passes both gates; safe for downstream embedding.
        "filtered_small_crop"   short side < min_crop_dim. Most often a backdrop /
                                background-collage face whose halftone texture
                                would otherwise score as sharp.
        "filtered_blurry"       quality score < blur_threshold (only checked
                                when the crop is large enough to trust the score).

    Size is checked before blur on purpose: small crops are noisy on the
    Laplacian variance metric (halftone print grain registers as edge content),
    so we don't trust their scores. Crops too small for the score to be
    meaningful get attributed to the size filter, not the blur filter.

    Passing min_crop_dim=0 disables the size gate; passing blur_threshold=0
    effectively disables the blur gate (any non-empty crop produces score > 0).
    """
    if min(width, height) < min_crop_dim:
        return "filtered_small_crop"
    if score < blur_threshold:
        return "filtered_blurry"
    return "ok"


def score_crop(crop_path: Path) -> dict:
    """Score a crop file on disk.

    Returns a dict with keys: path, width, height, score, error.
    `score` is None and `error` is populated when the file can't be decoded;
    callers should treat those rows as opaque (don't filter on them).
    """
    bgr = cv2.imread(str(crop_path))
    if bgr is None:
        return {
            "path": str(crop_path),
            "width": None,
            "height": None,
            "score": None,
            "error": "cv2.imread returned None",
        }
    h, w = bgr.shape[:2]
    return {
        "path": str(crop_path),
        "width": int(w),
        "height": int(h),
        "score": laplacian_variance(bgr),
        "error": None,
    }
