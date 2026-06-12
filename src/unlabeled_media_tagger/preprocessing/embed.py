"""ArcFace embedding extraction for a single BGR face crop.

Thin wrapper around DeepFace.represent that matches the policy used in
`pipeline/embed_faces.py::embed_faces_in_image` exactly so that notebook
embeddings are bit-identical to local-pipeline embeddings on the same input:

- ``detector_backend="skip"`` (we already have the cropped face)
- ``enforce_detection=False`` (don't raise on tricky crops)
- BGR → RGB conversion done with `cv2.cvtColor` (DeepFace expects RGB)
- Output coerced to ``list[float]`` so it serializes through JSON / CSV

Designed to be copy-pasted into the notebook helpers cell verbatim per the
project's self-containment rule. Keep this function and the relevant body of
`embed_faces_in_image` in lockstep — they are the same algorithm in two
locations.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


DEFAULT_EMBEDDING_MODEL = "ArcFace"


def embed_face_crop_bgr(
    bgr_crop: np.ndarray,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[list[float]]:
    """Compute an embedding vector for a single face crop in BGR layout.

    Returns ``None`` when DeepFace cannot produce an embedding (empty result
    list, internal failure, malformed crop). Callers should treat ``None`` as
    "skip this face for clustering" — same policy as ``embed_faces_in_image``.

    Note: the import of ``DeepFace`` is deferred to call time so that modules
    that only need the cluster algorithm (e.g. tests of cluster.py) don't pay
    the DeepFace import cost.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    if bgr_crop.shape[0] < 10 or bgr_crop.shape[1] < 10:
        return None

    from deepface import DeepFace

    rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    try:
        result = DeepFace.represent(
            img_path=rgb_crop,
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False,
        )
    except Exception:
        return None

    if not isinstance(result, list) or len(result) == 0:
        return None
    embedding = result[0].get("embedding")
    if not embedding:
        return None

    return [float(value) for value in embedding]
