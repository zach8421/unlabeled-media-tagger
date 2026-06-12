"""Tests for preprocessing/cluster.py.

The headline test is parity with `pipeline/compare.py::CompareStage` — if
`compare.py` ever drifts from `preprocessing/cluster.py` these tests catch it.
"""

from __future__ import annotations

import random

from unlabeled_media_tagger.pipeline.compare import CompareStage
from unlabeled_media_tagger.preprocessing.cluster import (
    cluster_face_records,
    cosine_similarity,
)


def test_cosine_similarity_basic():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_clusters_similar_embeddings_into_one_group():
    faces = [
        {"embedding": [1.0, 0.0], "drive_id": "a"},
        {"embedding": [0.99, 0.01], "drive_id": "b"},
        {"embedding": [0.0, 1.0], "drive_id": "c"},
    ]
    clusters = cluster_face_records(faces, similarity_threshold=0.9)
    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert len(clusters[1]) == 1


def test_each_face_carries_cluster_fields():
    faces = [
        {"embedding": [1.0, 0.0], "face_id": "x"},
        {"embedding": [0.0, 1.0], "face_id": "y"},
    ]
    clusters = cluster_face_records(faces, similarity_threshold=0.9)
    seed = clusters[0][0]
    assert seed["cluster_id"] == 0
    assert seed["cluster_label"] == "person_000"
    assert seed["similarity_to_cluster"] == 1.0
    assert seed["face_id"] == "x"


def test_skips_records_missing_embedding():
    faces = [
        {"embedding": [1.0, 0.0]},
        {"embedding": None},
        {"embedding": []},
        {"drive_id": "no-embedding"},
    ]
    clusters = cluster_face_records(faces, similarity_threshold=0.9)
    total = sum(len(rows) for rows in clusters.values())
    assert total == 1


def test_parity_with_compare_stage_on_random_data():
    """Same input → same partition as the pipeline's CompareStage.

    The clustering algorithm is order-dependent (each face is assigned by
    nearest existing centroid), so we test parity by feeding identical input
    sequences to both implementations and asserting identical cluster
    assignments per face index.
    """
    random.seed(42)
    faces = []
    base_vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    for i in range(60):
        base = base_vectors[i % 3]
        noise = [random.gauss(0, 0.05) for _ in range(3)]
        faces.append(
            {
                "embedding": [b + n for b, n in zip(base, noise)],
                "drive_id": f"file-{i}",
            }
        )

    stage = CompareStage(config={"similarity_threshold": 0.68})
    pipeline_clusters = stage.compare_faces(faces)
    notebook_clusters = cluster_face_records(faces, similarity_threshold=0.68)

    assert set(pipeline_clusters.keys()) == set(notebook_clusters.keys())
    for cluster_id in pipeline_clusters:
        pipeline_drive_ids = sorted(
            row["drive_id"] for row in pipeline_clusters[cluster_id]
        )
        notebook_drive_ids = sorted(
            row["drive_id"] for row in notebook_clusters[cluster_id]
        )
        assert pipeline_drive_ids == notebook_drive_ids


def test_low_threshold_collapses_everything_into_one_cluster():
    faces = [
        {"embedding": [1.0, 0.0]},
        {"embedding": [0.0, 1.0]},
        {"embedding": [-1.0, 0.0]},
    ]
    clusters = cluster_face_records(faces, similarity_threshold=-2.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3
