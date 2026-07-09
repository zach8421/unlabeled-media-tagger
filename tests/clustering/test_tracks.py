"""Behavior pins for within-file track collapse (clustering/tracks.py)."""

from __future__ import annotations

import numpy as np
import pytest

from unlabeled_media_tagger.clustering.tracks import Track, build_tracks_for_file


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _person_vectors(base_seed: int, count: int, jitter: float = 0.02):
    """Vectors tightly clustered around one random direction (one person)."""
    rng = np.random.default_rng(base_seed)
    base = rng.standard_normal(512)
    return np.stack([
        _unit(base + jitter * rng.standard_normal(512)) for _ in range(count)
    ])


def _rows(frames):
    return [{"frame_index": f, "quality_score": q}
            for f, q in frames]


def test_same_person_across_frames_merges_to_one_track():
    emb = _person_vectors(1, 4).astype(np.float32)
    rows = _rows([(0, 100), (150, 300), (300, 200), (450, 50)])
    tracks = build_tracks_for_file(rows, emb)
    assert len(tracks) == 1
    assert tracks[0].member_indices == [0, 1, 2, 3]
    assert tracks[0].min_intra_sim > 0.9


def test_two_distinct_people_never_merge():
    a = _person_vectors(1, 3)
    b = _person_vectors(2, 3)  # independent direction: near-zero similarity
    emb = np.concatenate([a, b]).astype(np.float32)
    rows = _rows([(0, 1), (150, 1), (300, 1), (0, 1), (150, 1), (300, 1)])
    tracks = build_tracks_for_file(rows, emb)
    assert len(tracks) == 2
    assert tracks[0].member_indices == [0, 1, 2]
    assert tracks[1].member_indices == [3, 4, 5]


def test_same_frame_faces_never_merge_even_if_similar():
    emb = _person_vectors(3, 2, jitter=0.001).astype(np.float32)  # sim ~1.0
    rows = _rows([(0, 1), (0, 1)])  # same frame -> different people
    tracks = build_tracks_for_file(rows, emb)
    assert len(tracks) == 2


def test_transitive_same_frame_blocking():
    """A merges B, then C (same frame as B) must not join the merged track."""
    emb = _person_vectors(4, 3, jitter=0.001).astype(np.float32)
    rows = _rows([(0, 1), (150, 1), (150, 1)])
    tracks = build_tracks_for_file(rows, emb)
    members = sorted(tuple(t.member_indices) for t in tracks)
    # Exactly one merge can happen; index 1 and 2 share frame 150.
    assert len(tracks) == 2
    assert ([0, 1] in [t.member_indices for t in tracks]
            or [0, 2] in [t.member_indices for t in tracks]), members


def test_excluded_index_stays_singleton():
    emb = _person_vectors(5, 3, jitter=0.001).astype(np.float32)
    rows = _rows([(0, 1), (150, 1), (300, 1)])
    tracks = build_tracks_for_file(rows, emb, exclude_indices={1})
    assert [t.member_indices for t in tracks] == [[0, 2], [1]]


def test_image_singleton_and_empty():
    assert build_tracks_for_file([], np.empty((0, 512), np.float32)) == []
    emb = _person_vectors(6, 1).astype(np.float32)
    tracks = build_tracks_for_file([{"frame_index": "", "quality_score": 5}], emb)
    assert len(tracks) == 1
    assert isinstance(tracks[0], Track)
    assert tracks[0].mean_intra_sim == 1.0
    assert tracks[0].representative_indices == [0]


def test_every_face_lands_in_exactly_one_track():
    rng = np.random.default_rng(7)
    people = [np.stack([_unit(rng.standard_normal(512))] * 1)
              for _ in range(10)]
    emb = np.concatenate(people).astype(np.float32)
    rows = _rows([(i * 150, 1) for i in range(10)])
    tracks = build_tracks_for_file(rows, emb)
    seen = sorted(i for t in tracks for i in t.member_indices)
    assert seen == list(range(10))


def test_representatives_are_top_quality_and_embedding_unit_norm():
    emb = _person_vectors(8, 5).astype(np.float32)
    rows = _rows([(0, 10), (150, 500), (300, 50), (450, 900), (600, 700)])
    tracks = build_tracks_for_file(rows, emb)
    assert len(tracks) == 1
    assert tracks[0].representative_indices == [3, 4, 1]  # 900, 700, 500
    assert tracks[0].embedding.shape == (512,)
    assert abs(float(np.linalg.norm(tracks[0].embedding)) - 1.0) < 1e-6


def test_misaligned_inputs_raise():
    with pytest.raises(ValueError):
        build_tracks_for_file([{"frame_index": 0}], np.zeros((2, 512), np.float32))


def test_determinism():
    rng = np.random.default_rng(9)
    emb = np.stack([_unit(rng.standard_normal(512)) for _ in range(30)]).astype(np.float32)
    rows = _rows([(i % 5 * 150, rng.integers(0, 1000)) for i in range(30)])
    first = build_tracks_for_file(rows, emb)
    second = build_tracks_for_file(rows, emb)
    assert [t.member_indices for t in first] == [t.member_indices for t in second]
    assert [t.medoid_index for t in first] == [t.medoid_index for t in second]
