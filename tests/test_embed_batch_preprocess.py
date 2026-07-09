"""Parity pins for scripts/run_embed_batch.py's preprocessing path.

The batch embedder must feed the model EXACTLY what DeepFace.represent's
detector_backend="skip" path feeds it for the same crop (that path is what
`embed_face_crop_bgr`, the parity oracle, uses). represent()'s two channel
flips cancel, so the model receives the array exactly as passed to represent;
`embed_face_crop_bgr` passes RGB. The batch path therefore does
BGR->RGB->resize_image and nothing else.

These tests import deepface.modules.preprocessing (pulls in TF) but never
build the model — safe without a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

cv2 = pytest.importorskip("cv2")
deepface_preprocessing = pytest.importorskip("deepface.modules.preprocessing")


def _random_bgr(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _represent_skip_path_input(bgr: np.ndarray) -> np.ndarray:
    """What represent(img_path=rgb, detector_backend="skip") feeds the model.

    Replicates representation.py: the oracle passes RGB; represent flips to
    "RGB" then flips back to "BGR" — a no-op pair — then resize_image with the
    (swapped) target size and normalize_input("base") (a no-op).
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = rgb[:, :, ::-1]          # represent: "Convert to RGB format"
    img = img[:, :, ::-1]          # represent: "rgb to bgr" (cancels)
    img = deepface_preprocessing.resize_image(img=img, target_size=(112, 112))
    return deepface_preprocessing.normalize_input(img=img, normalization="base")


class _FakeModel:
    input_shape = (112, 112)


@pytest.fixture()
def preprocess(tmp_path):
    from run_embed_batch import make_preprocessor

    return make_preprocessor(_FakeModel())


@pytest.mark.parametrize("shape", [(150, 120), (112, 112), (37, 301), (400, 400)])
def test_batch_preprocess_matches_represent_skip_path(preprocess, tmp_path, shape):
    bgr = _random_bgr(*shape, seed=shape[0])
    crop_path = tmp_path / "crop.jpg"
    cv2.imwrite(str(crop_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    decoded = cv2.imread(str(crop_path))  # compare post-JPEG, like production

    ours = preprocess(str(crop_path))
    expected = _represent_skip_path_input(decoded)

    assert ours is not None
    assert ours.shape == (1, 112, 112, 3)
    assert ours.dtype == np.float32
    np.testing.assert_array_equal(ours, expected)


def test_preprocess_rejects_missing_and_degenerate(preprocess, tmp_path):
    assert preprocess(str(tmp_path / "nope.jpg")) is None

    tiny = _random_bgr(8, 40)  # short side < 10, same guard as the oracle
    tiny_path = tmp_path / "tiny.jpg"
    cv2.imwrite(str(tiny_path), tiny)
    assert preprocess(str(tiny_path)) is None


def test_embed_arrays_pads_partial_batch_correctly():
    """Pad+slice of the tail batch must not leak padding into results."""
    from run_embed_batch import embed_arrays

    class _RecordingModel:
        input_shape = (112, 112)
        output_shape = 512
        seen_shapes: list = []

        def forward(self, batch):
            self.seen_shapes.append(batch.shape)
            # Echo a recognizable function of each input so slicing is checkable.
            flat = batch.reshape(batch.shape[0], -1)
            out = np.zeros((batch.shape[0], 512), dtype=np.float32)
            out[:, 0] = flat.sum(axis=1)
            return out.tolist()

    model = _RecordingModel()
    arrays = [np.full((1, 112, 112, 3), fill_value=i + 1, dtype=np.float32)
              for i in range(5)]
    out = embed_arrays(model, arrays, batch_size=4)

    assert out.shape == (5, 512)
    # Every forward call saw the fixed batch shape.
    assert set(model.seen_shapes) == {(4, 112, 112, 3)}
    expected_sums = [(i + 1) * 112 * 112 * 3 for i in range(5)]
    np.testing.assert_allclose(out[:, 0], expected_sums, rtol=1e-6)
