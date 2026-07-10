"""Pure HTML builders for the review server. No I/O — testable as strings.

Grid/card idioms follow scripts/blur_filter_tune.py. All user-visible strings
go through html.escape; face_keys ride in data- attributes for the vanilla-JS
mark actions (POSTs to /api/mark, see server.py).

Keyboard map (group pages): j/k or arrows move focus, x toggles remove on the
focused face, c confirms the group, n focuses the name box, u undoes the last
event shown in the sidebar.
"""

from __future__ import annotations

import html
import json
from urllib.parse import quote


def crop_src(face_key: str) -> str:
    """/crop/ URL for a face_key, percent-encoded.

    Crop names inherit Drive file stems, which really do contain '#', '%'
    and '?' (6,394 '#' names in FOLDER 2); html.escape alone would let the
    browser truncate the URL at the fragment.
    """
    return "/crop/" + quote(face_key, safe="/")

FACES_PER_PAGE = 120
CLUSTERS_PER_PAGE = 200

_BASE_STYLE = """
body { font-family: system-ui, sans-serif; margin: 16px; background: #fafafa;
       color: #222; }
a { color: #0645ad; text-decoration: none; } a:hover { text-decoration: underline; }
h1 { font-size: 19px; } .sub { color: #666; font-size: 13px; margin: 4px 0 14px; }
table { border-collapse: collapse; font-size: 14px; }
td, th { padding: 4px 10px; border-bottom: 1px solid #e5e5e5; text-align: left; }
tr:hover { background: #f0f4ff; }
.badge { border-radius: 9px; padding: 1px 8px; font-size: 11px; color: #fff; }
.badge.confirmed { background: #2a8a2a; } .badge.rejected { background: #a33; }
.badge.new { background: #888; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr));
        gap: 8px; margin-top: 12px; }
.card { background: #fff; border: 2px solid #ddd; border-radius: 8px;
        padding: 5px; text-align: center; cursor: pointer; }
.card img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; }
.card .m { font-size: 10px; color: #888; overflow: hidden;
           text-overflow: ellipsis; white-space: nowrap; }
.card.removed { border-color: #c00; background: #ffecec; }
.card.removed img { opacity: 0.45; }
.card.focused { outline: 3px solid #4a80ff; }
.bar { position: sticky; top: 0; background: #fffbe8; border: 1px solid #e5d9a0;
       border-radius: 8px; padding: 9px 12px; display: flex; gap: 10px;
       align-items: center; z-index: 5; }
.bar input[type=text] { padding: 4px 8px; border: 1px solid #bbb;
                        border-radius: 5px; width: 220px; }
button { padding: 5px 12px; border: 1px solid #999; border-radius: 6px;
         background: #fff; cursor: pointer; }
button.primary { background: #2a8a2a; color: #fff; border-color: #257425; }
button.danger { background: #fff; color: #a33; border-color: #a33; }
.events { margin-top: 22px; font-size: 12px; color: #555; }
.events li { margin: 2px 0; }
.pager { margin: 14px 0; }
.pair-row { display: flex; gap: 14px; align-items: center; background: #fff;
            border: 1px solid #ddd; border-radius: 8px; padding: 8px;
            margin: 8px 0; }
.pair-row img { width: 96px; height: 96px; object-fit: cover; border-radius: 4px; }
.exemplars img { width: 72px; height: 72px; margin-right: 4px; }
kbd { background: #eee; border-radius: 3px; padding: 0 5px; font-size: 11px; }
"""


def _page(title: str, body: str, script: str = "") -> str:
    return (f"<!doctype html><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{_BASE_STYLE}</style>{body}"
            f"<script>{script}</script>")


def _pager(base: str, page: int, n_pages: int) -> str:
    if n_pages <= 1:
        return ""
    sep = "&" if "?" in base else "?"
    prev_link = (f"<a href='{base}{sep}page={page - 1}'>&larr; prev</a>"
                 if page > 0 else "")
    next_link = (f"<a href='{base}{sep}page={page + 1}'>next &rarr;</a>"
                 if page < n_pages - 1 else "")
    return (f"<div class='pager'>{prev_link} page {page + 1} / {n_pages} "
            f"{next_link}</div>")


def render_index(run_id: str, clusters: list, review_status: dict,
                 page: int = 0, medoid_of: dict | None = None,
                 triage: list | None = None, n_triage: int = 0) -> str:
    """clusters: cluster_summary rows (dicts). review_status:
    cluster_id(str) -> ('confirmed', identity_name) | ('rejected', '').
    medoid_of: cluster_id -> representative face_key for the thumbnail.
    triage: when given (triage.csv rows: rank/section/reason/cluster_id),
    show only those clusters in triage rank order with a "why" column.
    n_triage: triage queue size, for the header link on the full index.

    Confirmed clusters get a checkbox; selecting 2+ enables "same person",
    which merges their identities in the label store (undoable)."""
    medoid_of = medoid_of or {}
    if triage is not None:
        triage_of = {t["cluster_id"]: t for t in triage}
        clusters = [c for c in clusters if c["cluster_id"] in triage_of]

        def sort_key(row):
            reviewed = row["cluster_id"] in review_status
            return (reviewed, int(triage_of[row["cluster_id"]]["rank"]))
        base = "/?view=triage"
    else:
        triage_of = {}

        def sort_key(row):
            status = review_status.get(row["cluster_id"], None)
            reviewed = status is not None
            return (reviewed, -(int(row["n_events"]) * int(row["n_faces"])))
        base = "/"

    ordered = sorted(clusters, key=sort_key)
    n_pages = max(1, -(-len(ordered) // CLUSTERS_PER_PAGE))
    page = max(0, min(page, n_pages - 1))
    visible = ordered[page * CLUSTERS_PER_PAGE:(page + 1) * CLUSTERS_PER_PAGE]

    n_reviewed = sum(1 for c in clusters
                     if c["cluster_id"] in review_status)
    rows = []
    for row in visible:
        status = review_status.get(row["cluster_id"])
        if status is None:
            badge = "<span class='badge new'>unreviewed</span>"
        elif status[0] == "confirmed":
            name = html.escape(status[1] or "(unnamed)")
            badge = f"<span class='badge confirmed'>{name}</span>"
        else:
            badge = "<span class='badge rejected'>rejected</span>"
        if row.get("degenerate_suspect") == "true":
            badge += (" <span class='badge rejected' title='implausibly "
                      "coherent across many files — likely a junk attractor, "
                      "not a person'>&#9888; degenerate?</span>")
        if row.get("sibling_export_suspect") == "true":
            badge += (" <span class='badge rejected' title='members are one "
                      "repeated frame across sibling export files — often a "
                      "graphic overlay, not a person'>&#9888; overlay?</span>")
        if row.get("static_face_suspect") == "true":
            badge += (" <span class='badge rejected' title='face embedding "
                      "frozen across 15+ seconds of footage — likely a "
                      "poster, mural, sketch, or displayed photo rather than "
                      "a live person'>&#9888; static/artwork?</span>")
        confirmed = status is not None and status[0] == "confirmed"
        checkbox = (f"<input type='checkbox' class='samesel' "
                    f"data-cid='{row['cluster_id']}'>" if confirmed else "")
        medoid = medoid_of.get(row["cluster_id"])
        thumb = (f"<img src='{html.escape(crop_src(medoid), quote=True)}' "
                 f"loading='lazy' style='width:44px;height:44px;"
                 f"object-fit:cover;border-radius:4px;vertical-align:middle'>"
                 if medoid else "")
        reason_cell = ""
        if triage is not None:
            entry = triage_of[row["cluster_id"]]
            reason_cell = (f"<td class='m'>{html.escape(entry['section'])}: "
                           f"{html.escape(entry['reason'][:80])}</td>")
        rows.append(
            f"<tr><td>{checkbox}</td><td>{thumb}</td>"
            f"<td><a href='/group/{row['cluster_id']}'>"
            f"{html.escape(row['cluster_label'])}</a></td>"
            f"<td>{row['n_faces']}</td><td>{row['n_nodes']}</td>"
            f"<td>{row['n_files']}</td><td>{row['n_events']}</td>"
            f"<td>{badge}</td>{reason_cell}"
            f"<td class='m'>{html.escape(row['example_events'][:90])}</td></tr>")

    if triage is not None:
        heading = f"Triage queue — run {html.escape(run_id)}"
        crumbs = (f"{len(clusters)} triage clusters, {n_reviewed} reviewed · "
                  f"<a href='/'>all clusters</a>")
        reason_th = "<th>why</th>"
    else:
        heading = f"Clusters — run {html.escape(run_id)}"
        triage_link = (f" · <a href='/?view=triage'>triage queue "
                       f"({n_triage})</a>" if n_triage else "")
        crumbs = (f"{len(clusters)} clusters, {n_reviewed} reviewed"
                  f"{triage_link} · <a href='/queue'>growth review queue</a>"
                  f" · <a href='/merges'>merge candidates</a>")
        reason_th = ""
    body = (
        f"<h1>{heading}</h1>"
        f"<div class='sub'>{crumbs} · tick 2+ confirmed "
        f"clusters and <button id='samebtn' disabled "
        f"onclick='mergeSelected()'>mark same person</button></div>"
        + _pager(base, page, n_pages)
        + "<table><tr><th></th><th></th><th>cluster</th><th>faces</th>"
          "<th>tracks</th><th>files</th><th>events</th><th>status</th>"
          f"{reason_th}<th>example events</th></tr>"
        + "".join(rows) + "</table>"
        + _pager(base, page, n_pages))
    return _page(f"Clusters — {run_id}", body, _INDEX_SCRIPT)


_INDEX_SCRIPT = """
function selected() {
  return Array.from(document.querySelectorAll('.samesel:checked'))
              .map(el => el.dataset.cid);
}
document.addEventListener('change', e => {
  if (e.target.classList.contains('samesel'))
    document.getElementById('samebtn').disabled = selected().length < 2;
});
async function mergeSelected() {
  const cids = selected();
  if (cids.length < 2) return;
  if (!confirm(`Merge identities of ${cids.length} clusters as one person?`))
    return;
  const res = await fetch('/api/mark', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'merge_cluster_identities',
                          cluster_ids: cids})});
  if (!res.ok) { alert('merge failed: ' + await res.text()); return; }
  location.reload();
}
"""


_GROUP_SCRIPT = """
const cards = () => Array.from(document.querySelectorAll('.card'));
let focus = -1;
function setFocus(i) {
  cards().forEach(c => c.classList.remove('focused'));
  if (i >= 0 && i < cards().length) {
    focus = i; const el = cards()[i];
    el.classList.add('focused');
    el.scrollIntoView({block: 'nearest'});
  }
}
async function api(payload) {
  const res = await fetch('/api/mark', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)});
  if (!res.ok) { alert('mark failed: ' + await res.text()); return false; }
  return true;
}
async function toggleRemove(card) {
  const removed = card.classList.contains('removed');
  const ok = await api({action: removed ? 'undo_remove_face' : 'remove_face',
                        cluster_id: CLUSTER_ID, face_key: card.dataset.fk});
  if (ok) location.reload();
}
async function confirmGroup() {
  const name = document.getElementById('name').value.trim();
  if (await api({action: 'confirm_cluster', cluster_id: CLUSTER_ID,
                 identity_name: name})) location.reload();
}
async function rejectGroup() {
  if (!window.confirm('Reject this whole cluster (not a coherent person)?'))
    return;
  if (await api({action: 'reject_cluster', cluster_id: CLUSTER_ID}))
    location.reload();
}
async function undoEvent(eventId) {
  if (await api({action: 'undo', event_id: eventId})) location.reload();
}
document.addEventListener('click', e => {
  const card = e.target.closest('.card');
  if (card) toggleRemove(card);
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'j' || e.key === 'ArrowRight') setFocus(Math.min(focus + 1, cards().length - 1));
  else if (e.key === 'k' || e.key === 'ArrowLeft') setFocus(Math.max(focus - 1, 0));
  else if (e.key === 'x' && focus >= 0) toggleRemove(cards()[focus]);
  else if (e.key === 'c') confirmGroup();
  else if (e.key === 'n') { document.getElementById('name').focus(); e.preventDefault(); }
  else if (e.key === 'u') { const b = document.querySelector('.events button'); if (b) b.click(); }
});
async function mergeSimilar() {
  const cids = Array.from(document.querySelectorAll('.simsel:checked'))
                    .map(el => el.dataset.cid);
  if (!cids.length) { alert('tick at least one confirmed similar cluster'); return; }
  if (!confirm(`Merge ${cids.length} cluster(s) into this one as the same person?`))
    return;
  if (await api({action: 'merge_cluster_identities',
                 cluster_ids: [CLUSTER_ID, ...cids]})) location.reload();
}
"""


def _status_badge(status: str | None, name: str = "") -> str:
    if status == "confirmed":
        return (f"<span class='badge confirmed'>"
                f"{html.escape(name or '(unnamed)')}</span>")
    if status == "rejected":
        return "<span class='badge rejected'>rejected</span>"
    return "<span class='badge new'>unreviewed</span>"


def _similar_strip(similar: list, confirmed: bool) -> str:
    """The merge-discovery strip: nearest clusters by centroid cosine.

    Checkboxes appear only for confirmed neighbors of a confirmed cluster —
    identity.merge is defined on confirmed identities, so anything else gets
    a link to go confirm first."""
    rows = []
    for s in similar:
        if s.get("merged"):  # already the same canonical identity
            checkbox = "<span class='badge confirmed'>merged</span>"
        else:
            checkbox = (f"<input type='checkbox' class='simsel' "
                        f"data-cid='{html.escape(str(s['cluster_id']), quote=True)}'>"
                        if confirmed and s["status"] == "confirmed" else "")
        thumb = (f"<img src='{html.escape(crop_src(s['medoid']), quote=True)}' "
                 f"loading='lazy'>" if s["medoid"] else "")
        try:
            shared = int(s["shared_events"] or 0)
        except ValueError:
            shared = 0
        shared_note = (f" · <b>{shared} shared event(s)</b>" if shared else "")
        rows.append(
            f"<div class='pair-row'>{checkbox}{thumb}"
            f"<div><a href='/group/{s['cluster_id']}'>"
            f"{html.escape(s['label'])}</a> "
            f"{_status_badge(s['status'], s['name'])}<br>"
            f"cos {float(s['cosine']):.3f} · {s['n_faces']} faces"
            f"{shared_note}</div></div>")
    action = ("<button onclick='mergeSimilar()'>merge selected into this "
              "cluster</button>" if confirmed else
              "<i>confirm this cluster to enable merging</i>")
    return (f"<div class='similar'><b>Similar clusters</b> — same person? "
            f"{action}{''.join(rows)}</div>")


def render_group(run_id: str, cluster_id: str, cluster_label: str,
                 faces: list, *, identity: dict | None, rejected: bool,
                 removed_keys: set, recent_events: list, page: int = 0,
                 sort: str = "central", similar: list | None = None) -> str:
    """faces: dicts with face_key, source_file_id, crop_file_name, track_id,
    similarity_to_cluster (may be ''). identity: {'identity_id','name'} when
    confirmed. recent_events: [{'event_id','type','summary'}] newest first.
    similar: neighbor suggestions (cluster_id, label, n_faces, cosine,
    shared_events, status, name, medoid) for the merge-discovery strip."""
    def sim(face):
        try:
            return float(face.get("similarity_to_cluster") or 0.0)
        except ValueError:
            return 0.0

    ordered = sorted(faces, key=sim, reverse=(sort == "central"))
    n_pages = max(1, -(-len(ordered) // FACES_PER_PAGE))
    page = max(0, min(page, n_pages - 1))
    visible = ordered[page * FACES_PER_PAGE:(page + 1) * FACES_PER_PAGE]

    if rejected:
        status = "<span class='badge rejected'>rejected</span>"
    elif identity:
        status = (f"<span class='badge confirmed'>"
                  f"{html.escape(identity['name'] or identity['identity_id'])}"
                  f"</span>")
    else:
        status = "<span class='badge new'>unreviewed</span>"

    other_sort = "outliers" if sort == "central" else "central"
    cards = []
    for face in visible:
        removed = face["face_key"] in removed_keys
        cards.append(
            f"<div class='card{' removed' if removed else ''}' "
            f"data-fk='{html.escape(face['face_key'], quote=True)}'>"
            f"<img src='{html.escape(crop_src(face['face_key']), quote=True)}' "
            f"loading='lazy'>"
            f"<div class='m'>{sim(face):.3f} · "
            f"{html.escape(face['crop_file_name'][-22:])}</div></div>")

    events_html = "".join(
        f"<li>{html.escape(e['summary'])} "
        f"<button onclick=\"undoEvent('{html.escape(e['event_id'], quote=True)}')\">"
        f"undo</button></li>"
        for e in recent_events)

    name_value = html.escape(identity["name"], quote=True) if identity else ""
    body = (
        f"<h1><a href='/'>&larr;</a> {html.escape(cluster_label)} {status}</h1>"
        f"<div class='sub'>{len(faces)} faces · sort: {sort} "
        f"(<a href='/group/{cluster_id}?sort={other_sort}'>switch to "
        f"{other_sort}-first</a>) · click or <kbd>x</kbd> = toggle remove, "
        f"<kbd>c</kbd> = confirm, <kbd>n</kbd> = name, <kbd>u</kbd> = undo"
        f"</div>"
        f"<div class='bar'>"
        f"<input type='text' id='name' placeholder='identity name (optional)' "
        f"value='{name_value}'>"
        f"<button class='primary' onclick='confirmGroup()'>Confirm all "
        f"(minus removed)</button>"
        f"<button class='danger' onclick='rejectGroup()'>Reject group</button>"
        f"<span>{len(removed_keys)} removed</span></div>"
        + (_similar_strip(similar,
                          confirmed=identity is not None and not rejected)
           if similar else "")
        + _pager(f"/group/{cluster_id}", page, n_pages)
        + f"<div class='grid'>{''.join(cards)}</div>"
        + _pager(f"/group/{cluster_id}", page, n_pages)
        + f"<div class='events'><b>Recent events</b><ul>{events_html}</ul></div>")
    script = f"const CLUSTER_ID = {json.dumps(cluster_id)};" + _GROUP_SCRIPT
    return _page(f"{cluster_label} — {run_id}", body, script)


_QUEUE_SCRIPT = """
async function api(payload) {
  const res = await fetch('/api/mark', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)});
  if (!res.ok) { alert('failed: ' + await res.text()); return false; }
  return true;
}
async function decide(btn, accept) {
  const row = btn.closest('.pair-row');
  const ok = await api({action: accept ? 'accept_proposal' : 'reject_proposal',
                        face_key: row.dataset.fk,
                        identity_id: row.dataset.iid});
  if (ok) { row.style.opacity = 0.35; btn.disabled = true; }
}
async function acceptAllAttach(btn) {
  btn.disabled = true;
  if (await api({action: 'accept_all_attach'})) location.reload();
  else btn.disabled = false;
}
"""


def render_queue(run_id: str, proposals: list, decided_keys: set) -> str:
    """proposals: dicts with face_key, identity_id, identity_name, score,
    margin, exemplar_face_keys (list), decision ('attach'|'review').
    decided_keys: face_keys already accepted/rejected (dimmed, no buttons)."""
    attach = [p for p in proposals if p.get("decision") == "attach"]
    review = [p for p in proposals if p.get("decision") != "attach"]
    n_attach_pending = sum(1 for p in attach
                           if p["face_key"] not in decided_keys)
    sections = []
    if attach:
        sections.append(
            f"<h2>High-confidence attaches ({len(attach)}, "
            f"{n_attach_pending} pending)</h2>"
            f"<button class='primary' onclick='acceptAllAttach(this)'>"
            f"Accept all {n_attach_pending} pending</button>")
        sections.extend(_proposal_row(p, p["face_key"] in decided_keys)
                        for p in attach)
    sections.append(f"<h2>Gray zone — human call ({len(review)})</h2>")
    sections.extend(_proposal_row(p, p["face_key"] in decided_keys)
                    for p in review)

    body = (f"<h1><a href='/'>&larr;</a> Growth review queue — run "
            f"{html.escape(run_id)}</h1>"
            f"<div class='sub'>{len(proposals)} proposals; accept attaches "
            f"the whole track, reject records a cannot-link so it is never "
            f"proposed again</div>" + "".join(sections))
    return _page(f"Queue — {run_id}", body, _QUEUE_SCRIPT)


MERGES_PER_PAGE = 100

_MERGES_SCRIPT = """
async function mergePair(btn, a, b) {
  if (!confirm('Merge these two identities as one person?')) return;
  btn.disabled = true;
  const res = await fetch('/api/mark', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'merge_cluster_identities',
                          cluster_ids: [a, b]})});
  if (!res.ok) { alert('merge failed: ' + await res.text());
                 btn.disabled = false; return; }
  location.reload();
}
"""


def render_merges(run_id: str, pairs: list, page: int = 0) -> str:
    """Corpus-wide merge candidates: both-confirmed cluster pairs by
    descending centroid cosine. pairs: dicts with cosine, shared_events,
    merged (already same canonical identity), clusters=[{cluster_id, label,
    n_faces, name, medoid} x2]."""
    n_pages = max(1, -(-len(pairs) // MERGES_PER_PAGE))
    page = max(0, min(page, n_pages - 1))
    visible = pairs[page * MERGES_PER_PAGE:(page + 1) * MERGES_PER_PAGE]

    n_open = sum(1 for p in pairs if not p["merged"])
    rows = []
    for pair in visible:
        sides = []
        for c in pair["clusters"]:
            thumb = (f"<img src='{html.escape(crop_src(c['medoid']), quote=True)}' "
                     f"loading='lazy'>" if c["medoid"] else "")
            sides.append(
                f"{thumb}<div><a href='/group/{c['cluster_id']}'>"
                f"{html.escape(c['label'])}</a> "
                f"<span class='badge confirmed'>"
                f"{html.escape(c['name'] or '(unnamed)')}</span><br>"
                f"{c['n_faces']} faces</div>")
        try:
            shared = int(pair["shared_events"] or 0)
        except ValueError:
            shared = 0
        shared_note = (f" · <b>{shared} shared event(s)</b>" if shared else "")
        a, b = (c["cluster_id"] for c in pair["clusters"])
        action = ("<span class='badge confirmed'>merged</span>"
                  if pair["merged"] else
                  f"<button class='primary' onclick=\"mergePair(this, "
                  f"'{html.escape(str(a), quote=True)}', "
                  f"'{html.escape(str(b), quote=True)}')\">"
                  f"same person</button>")
        rows.append(
            f"<div class='pair-row' "
            f"style='{'opacity:.4' if pair['merged'] else ''}'>"
            + "".join(sides)
            + f"<div>cos {float(pair['cosine']):.3f}{shared_note}</div>"
            + f"<div>{action}</div></div>")

    if not pairs:
        rows.append("<p><i>No both-confirmed similar pairs yet — confirm "
                    "more clusters, or regenerate cluster_neighbors.csv "
                    "with a lower --min-cos.</i></p>")
    body = (f"<h1><a href='/'>&larr;</a> Merge candidates — run "
            f"{html.escape(run_id)}</h1>"
            f"<div class='sub'>{len(pairs)} both-confirmed pairs by centroid "
            f"similarity, {n_open} still separate · merging pools members "
            f"for growth and is undoable per pair</div>"
            + _pager("/merges", page, n_pages)
            + "".join(rows)
            + _pager("/merges", page, n_pages))
    return _page(f"Merges — {run_id}", body, _MERGES_SCRIPT)


def _proposal_row(prop: dict, decided: bool) -> str:
    n_members = len(prop.get("member_face_keys") or []) or 1
    exemplars = "".join(
        f"<img src='{html.escape(crop_src(k), quote=True)}' loading='lazy'>"
        for k in prop["exemplar_face_keys"][:5])
    buttons = ("<i>decided</i>" if decided else
               "<button class='primary' onclick='decide(this, true)'>"
               "Accept</button> "
               "<button class='danger' onclick='decide(this, false)'>"
               "Not this person</button>")
    return (
        f"<div class='pair-row' style='{'opacity:.4' if decided else ''}' "
        f"data-fk='{html.escape(prop['face_key'], quote=True)}' "
        f"data-iid='{html.escape(prop['identity_id'], quote=True)}'>"
        f"<img src='{html.escape(crop_src(prop['face_key']), quote=True)}' "
        f"loading='lazy'>"
        f"<div>&rarr; <b>"
        f"{html.escape(prop['identity_name'] or prop['identity_id'])}"
        f"</b><br>score {prop['score']:.3f} · margin {prop['margin']:.3f} · "
        f"{n_members} face(s)</div>"
        f"<div class='exemplars'>{exemplars}</div>"
        f"<div>{buttons}</div></div>")
