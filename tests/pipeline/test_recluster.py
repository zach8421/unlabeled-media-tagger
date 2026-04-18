"""Tests for offline face reclustering."""

from unlabeled_media_tagger.pipeline.recluster import recluster_faces


def test_recluster_groups_similar_faces_without_model_calls():
    """Test that reclustering groups records from existing embeddings."""
    faces = [
        {"embedding": [1.0, 0.0], "frame_path": "a.jpg"},
        {"embedding": [0.99, 0.01], "frame_path": "b.jpg"},
        {"embedding": [0.0, 1.0], "frame_path": "c.jpg"},
    ]

    clusters = recluster_faces(
        faces,
        edge_threshold=0.9,
        merge_min_similarity=0.9,
        merge_mean_similarity=0.9,
    )

    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert len(clusters[1]) == 1


def test_recluster_prevents_same_frame_merge():
    """Test that two faces from the same frame are not merged by default."""
    faces = [
        {"embedding": [1.0, 0.0], "frame_path": "same-frame.jpg"},
        {"embedding": [0.99, 0.01], "frame_path": "same-frame.jpg"},
    ]

    clusters = recluster_faces(
        faces,
        edge_threshold=0.9,
        merge_min_similarity=0.9,
        merge_mean_similarity=0.9,
    )

    assert len(clusters) == 2


def test_recluster_can_allow_same_frame_merge():
    """Test that same-frame merge prevention is configurable."""
    faces = [
        {"embedding": [1.0, 0.0], "frame_path": "same-frame.jpg"},
        {"embedding": [0.99, 0.01], "frame_path": "same-frame.jpg"},
    ]

    clusters = recluster_faces(
        faces,
        edge_threshold=0.9,
        merge_min_similarity=0.9,
        merge_mean_similarity=0.9,
        prevent_same_frame_merge=False,
    )

    assert len(clusters) == 1
