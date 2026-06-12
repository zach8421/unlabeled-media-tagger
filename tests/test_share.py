"""Tests for preprocessing/share.py."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from unlabeled_media_tagger.preprocessing.share import (
    CONTACT_SHEET_DIR_NAME,
    FACE_CLUSTERS_COLUMNS,
    SUMMARY_COLUMNS,
    copy_crop_to_contact_sheet,
    flatten_clusters_to_face_rows,
    group_rows_by_cluster,
    select_representative_face,
    summarize_clusters,
)


def _face(**overrides):
    base = {
        "source_file_id": "file-A",
        "source_file_name": "shot.jpg",
        "face_index": 0,
        "bbox_x": 10,
        "bbox_y": 20,
        "bbox_w": 100,
        "bbox_h": 120,
        "confidence": 0.99,
        "crop_path": "/tmp/crop.jpg",
        "cluster_id": 0,
        "cluster_label": "person_000",
        "similarity_to_cluster": 1.0,
    }
    base.update(overrides)
    return base


def test_flatten_emits_pipeline_schema_in_order():
    clusters = {0: [_face()]}
    rows = flatten_clusters_to_face_rows(clusters)
    assert list(rows[0].keys()) == FACE_CLUSTERS_COLUMNS
    assert rows[0]["drive_id"] == "file-A"
    assert rows[0]["media_name"] == "shot.jpg"
    assert rows[0]["frame_path"] == "/tmp/crop.jpg"
    assert rows[0]["timestamp_sec"] == ""
    assert rows[0]["frame_index"] == ""
    assert rows[0]["model_name"] == "ArcFace"
    assert rows[0]["detector_backend"] == "retinaface"


def test_flatten_sorts_by_cluster_id():
    clusters = {
        2: [_face(cluster_id=2, cluster_label="person_002")],
        0: [_face(cluster_id=0, cluster_label="person_000")],
        1: [_face(cluster_id=1, cluster_label="person_001")],
    }
    rows = flatten_clusters_to_face_rows(clusters)
    cluster_ids = [row["cluster_id"] for row in rows]
    assert cluster_ids == [0, 1, 2]


def test_flatten_uses_explicit_model_args():
    clusters = {0: [_face()]}
    rows = flatten_clusters_to_face_rows(
        clusters,
        model_name="Facenet",
        detector_backend="mtcnn",
    )
    assert rows[0]["model_name"] == "Facenet"
    assert rows[0]["detector_backend"] == "mtcnn"


def test_select_representative_picks_highest_similarity():
    rows = [
        _face(face_index=0, similarity_to_cluster=0.7, bbox_w=50, bbox_h=50),
        _face(face_index=1, similarity_to_cluster=0.99, bbox_w=20, bbox_h=20),
        _face(face_index=2, similarity_to_cluster=0.5, bbox_w=200, bbox_h=200),
    ]
    rep = select_representative_face(rows)
    assert rep["face_index"] == 1


def test_select_representative_breaks_ties_by_area_then_frame():
    rows = [
        _face(face_index=2, similarity_to_cluster=0.9, bbox_w=100, bbox_h=100),
        _face(face_index=0, similarity_to_cluster=0.9, bbox_w=100, bbox_h=100),
        _face(face_index=1, similarity_to_cluster=0.9, bbox_w=200, bbox_h=200),
    ]
    rep = select_representative_face(rows)
    assert rep["face_index"] == 1  # largest area wins

    rows_distinct_frame = [
        _face(face_index=2, frame_index=5, similarity_to_cluster=0.9, bbox_w=100, bbox_h=100),
        _face(face_index=0, frame_index=2, similarity_to_cluster=0.9, bbox_w=100, bbox_h=100),
    ]
    rep_distinct = select_representative_face(rows_distinct_frame)
    assert rep_distinct["frame_index"] == 2  # lower frame_index wins


def test_select_representative_returns_none_for_empty():
    assert select_representative_face([]) is None


def test_group_rows_by_cluster_sorts_labels():
    rows = [
        _face(cluster_label="person_002"),
        _face(cluster_label="person_000"),
        _face(cluster_label="person_001"),
    ]
    grouped = group_rows_by_cluster(rows)
    assert list(grouped.keys()) == ["person_000", "person_001", "person_002"]


def test_summarize_clusters_sorts_by_face_count_desc():
    grouped = {
        "person_000": [_face(cluster_id=0)] * 3,
        "person_001": [_face(cluster_id=1)] * 7,
        "person_002": [_face(cluster_id=2)] * 1,
    }
    summaries = summarize_clusters(grouped)
    counts = [s["face_count"] for s in summaries]
    assert counts == [7, 3, 1]
    assert list(summaries[0].keys()) == SUMMARY_COLUMNS


def test_summarize_clusters_uses_url_when_provided():
    grouped = {"person_000": [_face()]}
    url_by_label = {"person_000": "https://drive.google.com/uc?export=view&id=XYZ"}
    summaries = summarize_clusters(grouped, url_by_label=url_by_label)
    assert summaries[0]["contact_sheet"] == url_by_label["person_000"]


def test_summarize_clusters_falls_back_to_local_path():
    grouped = {"person_000": [_face()]}
    summaries = summarize_clusters(grouped)
    assert summaries[0]["contact_sheet"] == f"{CONTACT_SHEET_DIR_NAME}/person_000.jpg"


def test_copy_crop_to_contact_sheet_handles_missing_source(tmp_path: Path):
    dest = tmp_path / "out.jpg"
    assert not copy_crop_to_contact_sheet("", dest)
    assert not copy_crop_to_contact_sheet("/nonexistent/path.jpg", dest)
    assert not dest.exists()


def test_copy_crop_to_contact_sheet_resizes_large_crop(tmp_path: Path):
    src = tmp_path / "src.jpg"
    Image.fromarray(np.zeros((1200, 800, 3), dtype=np.uint8), mode="RGB").save(
        src, "JPEG", quality=90
    )
    dest = tmp_path / "out.jpg"
    ok = copy_crop_to_contact_sheet(str(src), dest, max_dim=512)
    assert ok
    assert dest.exists()
    out = Image.open(dest)
    assert max(out.width, out.height) == 512


def test_copy_crop_to_contact_sheet_preserves_small_crop(tmp_path: Path):
    src = tmp_path / "src.jpg"
    Image.fromarray(np.zeros((200, 150, 3), dtype=np.uint8), mode="RGB").save(
        src, "JPEG", quality=90
    )
    dest = tmp_path / "out.jpg"
    ok = copy_crop_to_contact_sheet(str(src), dest, max_dim=512)
    assert ok
    out = Image.open(dest)
    assert out.width == 150 and out.height == 200
