"""Sanity tests for blur scoring. Synthetic crops only."""

from __future__ import annotations

import cv2
import numpy as np

from unlabeled_media_tagger.preprocessing.blur import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_MIN_CROP_DIM,
    classify_quality,
    laplacian_variance,
)


def _checkerboard(size: int = 200, square: int = 20) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if ((y // square) + (x // square)) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


def test_sharp_scores_higher_than_blurry():
    sharp = _checkerboard()
    mid = cv2.GaussianBlur(sharp, (0, 0), sigmaX=2)
    very_blurry = cv2.GaussianBlur(sharp, (0, 0), sigmaX=8)

    s = laplacian_variance(sharp)
    m = laplacian_variance(mid)
    b = laplacian_variance(very_blurry)

    assert s > m > b, f"expected sharp > mid > blurry, got {s} > {m} > {b}"


def test_empty_returns_zero():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert laplacian_variance(empty) == 0.0


def test_returns_python_float():
    assert isinstance(laplacian_variance(_checkerboard()), float)


def test_classify_quality_size_filter_runs_before_blur():
    # Tiny but high-scoring crop (the halftone-backdrop failure mode):
    # without the size gate, this would be "ok"; with it, it's filtered.
    assert classify_quality(
        score=200.0, width=50, height=50,
        blur_threshold=45.0, min_crop_dim=100,
    ) == "filtered_small_crop"


def test_classify_quality_blurry_large_crop():
    assert classify_quality(
        score=10.0, width=300, height=400,
        blur_threshold=45.0, min_crop_dim=100,
    ) == "filtered_blurry"


def test_classify_quality_ok():
    assert classify_quality(
        score=80.0, width=300, height=400,
        blur_threshold=45.0, min_crop_dim=100,
    ) == "ok"


def test_classify_quality_uses_short_side():
    # Tall but narrow crop should be size-filtered if the short side is below.
    assert classify_quality(
        score=200.0, width=50, height=1000,
        blur_threshold=45.0, min_crop_dim=100,
    ) == "filtered_small_crop"


def test_classify_quality_defaults_match_tuned_values():
    assert DEFAULT_BLUR_THRESHOLD == 45.0
    assert DEFAULT_MIN_CROP_DIM == 100
