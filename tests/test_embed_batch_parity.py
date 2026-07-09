"""GPU parity test: batch embedding path vs the single-image oracle.

Slow (builds ArcFace, needs the GPU stack) and requires real FOLDER 2 crops,
so it is opt-in:  RUN_GPU_TESTS=1 ./.venv/bin/python -m pytest \
    tests/test_embed_batch_parity.py

The same gate runs operationally as `run_embed_batch.py --self-check N`
before any full embed run (passed 2026-07-06: min cosine 1.000000 over 1,000
real crops, max component delta 8.3e-07).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CROPS_ROOT = Path("/mnt/media1/folder2_detect/crops")

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_GPU_TESTS"),
    reason="GPU parity test; set RUN_GPU_TESTS=1 to run",
)


def test_batch_of_eight_matches_single_path_on_real_crops():
    if not CROPS_ROOT.is_dir():
        pytest.skip("FOLDER 2 crops not available on this machine")
    import cv2

    from run_embed_batch import build_model, embed_arrays, make_preprocessor
    from unlabeled_media_tagger.preprocessing.embed import embed_face_crop_bgr

    crop_paths = []
    for file_dir in sorted(CROPS_ROOT.iterdir()):
        for crop in sorted(file_dir.glob("*.jpg"))[:2]:
            crop_paths.append(crop)
        if len(crop_paths) >= 8:
            break
    assert len(crop_paths) >= 8, "not enough crops on disk"
    crop_paths = crop_paths[:8]

    model = build_model()
    preprocess = make_preprocessor(model)

    arrays, oracle = [], []
    for path in crop_paths:
        arr = preprocess(str(path))
        single = embed_face_crop_bgr(cv2.imread(str(path)))
        assert arr is not None and single is not None, path
        arrays.append(arr)
        oracle.append(single)

    batch = embed_arrays(model, arrays, batch_size=8)
    batch /= np.linalg.norm(batch, axis=1, keepdims=True)
    reference = np.asarray(oracle, dtype=np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)

    cosines = np.sum(batch * reference, axis=1)
    assert cosines.min() >= 0.9999, cosines
