"""People export: aggregation, ordering, merges/retired, gallery HTML."""

from __future__ import annotations

from unlabeled_media_tagger.review.export import (
    collect_people,
    render_gallery,
)
from unlabeled_media_tagger.review.store import State


def _state():
    state = State()
    state.identities = {
        "identity_0001": {"name": "Ana <A>"},
        "identity_0002": {"name": ""},
        "identity_0003": {"name": "Old"},   # merged away into 0001
        "identity_0004": {"name": "Gone", "retired": True},
    }
    state.merged_into = {"identity_0003": "identity_0001"}
    for face_key, identity in [
        ("fileA/a1.jpg", "identity_0001"),
        ("fileA/a2.jpg", "identity_0001"),
        ("fileB/b1.jpg", "identity_0003"),   # resolves to 0001 via merge
        ("fileC/c1.jpg", "identity_0002"),
        ("fileD/d1.jpg", "identity_0004"),   # retired: dropped
    ]:
        state.assignments[face_key] = {"identity_id": identity}
    return state


FILE_META = {
    "fileA": {"path": "EVENT ONE/clip.mp4", "media_type": "video"},
    "fileB": {"path": "EVENT TWO/photo.jpg", "media_type": "image"},
    "fileC": {"path": "EVENT ONE/other.mp4", "media_type": "video"},
}


def test_collect_people_aggregates_through_merges():
    people, person_files, sample_faces = collect_people(_state(), FILE_META)

    assert [p["identity_id"] for p in people] == \
        ["identity_0001", "identity_0002"]  # by n_files desc; retired gone
    ana = people[0]
    assert ana["name"] == "Ana <A>"
    assert ana["n_faces"] == 3 and ana["n_files"] == 2 and ana["n_events"] == 2
    assert "EVENT ONE" in ana["example_events"]

    ana_files = [r for r in person_files
                 if r["identity_id"] == "identity_0001"]
    assert [r["source_file_id"] for r in ana_files] == ["fileA", "fileB"]
    assert ana_files[0]["n_faces_in_file"] == 2
    assert ana_files[0]["drive_link"].endswith("/fileA/view")
    # gallery samples: one face per distinct file
    assert sample_faces["identity_0001"] == ["fileA/a1.jpg", "fileB/b1.jpg"]


def test_min_files_filter_and_unknown_meta():
    state = _state()
    state.assignments["fileX/x1.jpg"] = {"identity_id": "identity_0002"}
    people, person_files, _ = collect_people(state, FILE_META, min_files=2)
    assert [p["identity_id"] for p in people] == \
        ["identity_0001", "identity_0002"]
    unknown = [r for r in person_files if r["source_file_id"] == "fileX"][0]
    assert unknown["file_path"] == "" and unknown["media_type"] == ""

    people, _, _ = collect_people(_state(), FILE_META, min_files=2)
    assert [p["identity_id"] for p in people] == ["identity_0001"]


def test_gallery_html_escapes_and_marks_unnamed():
    people, _, sample_faces = collect_people(_state(), FILE_META)
    thumbs = {"fileA/a1.jpg": "data:image/jpeg;base64,AAAA",
              "fileB/b1.jpg": None}  # missing crop: img omitted, no crash
    page = render_gallery(people, sample_faces, thumbs,
                          title="People <index>", generated_at="2026-07-10")
    assert "Ana &lt;A&gt;" in page and "People &lt;index&gt;" in page
    assert page.count("<img") == 1  # only the loadable thumb
    assert "Who is this?" in page and "class='person unnamed'" in page
    assert "1 still unnamed" in page
