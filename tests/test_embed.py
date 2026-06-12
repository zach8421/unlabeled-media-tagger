"""Tests for preprocessing/embed.py.

Fast tests mock DeepFace.represent to exercise the wrapper's contract:
None-on-empty, None-on-too-small, type coercion, deferred-import behavior.

The deeper parity guarantee — that ``embed_face_crop_bgr`` produces a
bit-identical embedding to ``pipeline/embed_faces.py::embed_faces_in_image``
on the same in-memory BGR crop — is verified end-to-end during the notebook /
pipeline verification recipe (Colab run → diff), since both implementations
share the same DeepFace.represent invocation parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from unlabeled_media_tagger.preprocessing import embed as embed_module
from unlabeled_media_tagger.preprocessing.embed import embed_face_crop_bgr


def _fake_deepface(embedding_value):
    """Build a stub `deepface` module with a represent() returning embedding_value."""

    class _StubDeepFace:
        @staticmethod
        def represent(img_path, model_name, detector_backend, enforce_detection):
            if embedding_value is None:
                return []
            return [{"embedding": embedding_value}]

    class _StubModule:
        DeepFace = _StubDeepFace

    return _StubModule


def test_returns_none_for_none_input():
    assert embed_face_crop_bgr(None) is None


def test_returns_none_for_empty_array():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert embed_face_crop_bgr(empty) is None


def test_returns_none_for_too_small_crop():
    too_small = np.zeros((5, 5, 3), dtype=np.uint8)
    assert embed_face_crop_bgr(too_small) is None


def test_returns_list_of_floats_when_deepface_succeeds(monkeypatch):
    stub = _fake_deepface([1, 2, 3])
    monkeypatch.setitem(
        __import__("sys").modules, "deepface", stub
    )
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    result = embed_face_crop_bgr(crop)
    assert result == [1.0, 2.0, 3.0]
    assert all(isinstance(value, float) for value in result)


def test_returns_none_when_deepface_returns_empty_list(monkeypatch):
    stub = _fake_deepface(None)
    monkeypatch.setitem(
        __import__("sys").modules, "deepface", stub
    )
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    assert embed_face_crop_bgr(crop) is None


def test_returns_none_when_deepface_raises(monkeypatch):
    class _RaisingDeepFace:
        @staticmethod
        def represent(*args, **kwargs):
            raise RuntimeError("boom")

    class _StubModule:
        DeepFace = _RaisingDeepFace

    monkeypatch.setitem(
        __import__("sys").modules, "deepface", _StubModule
    )
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    assert embed_face_crop_bgr(crop) is None


def test_returns_none_when_deepface_returns_empty_embedding(monkeypatch):
    stub = _fake_deepface([])
    monkeypatch.setitem(
        __import__("sys").modules, "deepface", stub
    )
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    assert embed_face_crop_bgr(crop) is None


def test_default_model_constant():
    assert embed_module.DEFAULT_EMBEDDING_MODEL == "ArcFace"
