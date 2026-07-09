"""Triage-queue selection: determinism, section priority, budgets, strata."""

from __future__ import annotations

from unlabeled_media_tagger.review.triage import build_triage


def _row(cid, *, nodes=3, faces=10, files=3, events=2, sim=0.95,
         degenerate=False, sibling=False, static=False):
    return {
        "cluster_id": str(cid), "cluster_label": f"person_{cid:05d}",
        "n_nodes": str(nodes), "n_faces": str(faces), "n_files": str(files),
        "n_events": str(events), "mean_similarity": str(sim),
        "degenerate_suspect": str(degenerate).lower(),
        "sibling_export_suspect": str(sibling).lower(),
        "static_face_suspect": str(static).lower(),
        "example_events": f"ev{cid}",
    }


def _corpus():
    rows = []
    rows.append(_row(0, static=True))
    rows.append(_row(1, degenerate=True, sim=0.99, files=80))
    # sibling flags: 2 in the low-sim tail, 8 spread over strata
    rows.append(_row(2, sibling=True, sim=0.96, nodes=5, events=1))
    rows.append(_row(3, sibling=True, sim=0.955, nodes=9, events=3))
    for i in range(4, 12):
        rows.append(_row(i, sibling=True, sim=0.99,
                         nodes=5 + (i % 2) * 5, events=1 + (i % 3)))
    # bulk: varying sizes
    for i in range(12, 112):
        rows.append(_row(i, faces=5 + (i % 40), events=1 + (i % 4)))
    rows.append(_row(112, faces=900, events=9))   # clear largest
    return rows


def test_deterministic_and_deduped():
    rows = _corpus()
    a = build_triage(rows, seed=7)
    b = build_triage(rows, seed=7)
    assert [r["cluster_id"] for r in a] == [r["cluster_id"] for r in b]
    ids = [r["cluster_id"] for r in a]
    assert len(ids) == len(set(ids))
    assert [int(r["rank"]) for r in a] == list(range(1, len(a) + 1))


def test_sections_and_budgets():
    rows = _corpus()
    triage = build_triage(rows, top_faces=5, top_events=3, n_random=10,
                          n_size_weighted=4, sibling_sample=6, seed=0)
    by_section = {}
    for row in triage:
        by_section.setdefault(row["section"], []).append(row)

    assert [r["cluster_id"] for r in by_section["suspect_static"]] == ["0"]
    assert [r["cluster_id"] for r in by_section["suspect_degenerate"]] == ["1"]
    # low-sim tail always included ahead of the strata sample
    sibling_ids = {r["cluster_id"] for r in by_section["suspect_sibling"]}
    assert {"2", "3"} <= sibling_ids
    assert len(by_section["suspect_sibling"]) <= 6
    assert len(by_section["largest_faces"]) == 5
    assert by_section["largest_faces"][0]["cluster_id"] == "112"
    assert len(by_section["largest_events"]) == 3
    assert len(by_section["random_uniform"]) == 10
    assert len(by_section["random_size_weighted"]) == 4

    # section order is the review order
    order = [r["section"] for r in triage]
    assert order == sorted(
        order, key=["suspect_static", "suspect_degenerate", "suspect_sibling",
                    "largest_faces", "largest_events", "random_uniform",
                    "random_size_weighted"].index)


def test_small_corpus_does_not_overdraw():
    rows = [_row(i) for i in range(5)]
    triage = build_triage(rows, top_faces=50, top_events=20, n_random=100,
                          n_size_weighted=25, sibling_sample=60, seed=0)
    assert len(triage) == 5  # every cluster picked exactly once


def test_sibling_tail_sampled_when_over_budget():
    rows = [_row(i, sibling=True, sim=0.95, nodes=5) for i in range(20)]
    triage = build_triage(rows, sibling_sample=8, n_random=0,
                          n_size_weighted=0, top_faces=0, top_events=0,
                          seed=3)
    assert len(triage) == 8
    assert all(r["section"] == "suspect_sibling" for r in triage)
