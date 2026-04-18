"""Tests for compare stage."""

from unlabeled_media_tagger.pipeline.compare import CompareStage


def test_compare_stage_initialization():
    """Test that CompareStage can be initialized."""
    stage = CompareStage()
    assert stage is not None
    assert isinstance(stage.config, dict)


def test_compare_faces_clusters_similar_embeddings():
    """Test that compare_faces clusters embeddings by cosine similarity."""
    stage = CompareStage(config={"similarity_threshold": 0.9})
    clusters = stage.compare_faces(
        [
            {"embedding": [1.0, 0.0], "drive_id": "file-1"},
            {"embedding": [0.99, 0.01], "drive_id": "file-2"},
            {"embedding": [0.0, 1.0], "drive_id": "file-3"},
        ]
    )

    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert len(clusters[1]) == 1


def test_build_face_database_summarizes_clusters():
    """Test that build_face_database returns per-cluster summaries."""
    stage = CompareStage()
    database = stage.build_face_database(
        {
            0: [
                {"drive_id": "file-1", "media_name": "a.jpg"},
                {"drive_id": "file-2", "media_name": "b.jpg"},
            ]
        }
    )

    assert database[0]["cluster_label"] == "person_000"
    assert database[0]["face_count"] == 2
    assert database[0]["drive_file_ids"] == ["file-1", "file-2"]
