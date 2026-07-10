"""Sponsor-facing people export: who appears where, plus a naming gallery.

Turns the label store into three artifacts a non-technical sponsor can use:
  people.csv        one row per identity (coverage summary)
  person_files.csv  identity x file long table with Drive links
  people_gallery.html  self-contained visual index (embedded thumbnails),
                       ordered by coverage — the "who is this?" feedback
                       loop that gets identities named.

Pure data-shaping here; the CLI (scripts/build_people_export.py) wires the
label store, the detect manifest (file_id -> Drive path), and the cv2
thumbnail loader. Events are Drive-path dirnames (one folder = one shoot).
"""

from __future__ import annotations

import html
import posixpath

GALLERY_THUMBS_PER_PERSON = 4


def collect_people(state, file_meta: dict, *, min_files: int = 1):
    """-> (people, person_files, sample_faces).

    state: replayed LabelStore state. file_meta: source_file_id ->
    {"path", "media_type"}. people/person_files are csv-ready dict rows;
    sample_faces maps identity_id -> up to GALLERY_THUMBS_PER_PERSON
    face_keys drawn from distinct files (diverse looks aid recognition).
    """
    faces_of: dict = {}
    for face_key in sorted(state.assignments):
        identity_id = state.canonical_identity(
            state.assignments[face_key]["identity_id"])
        if state.identities.get(identity_id, {}).get("retired"):
            continue
        faces_of.setdefault(identity_id, []).append(face_key)

    people, person_files, sample_faces = [], [], {}
    for identity_id, face_keys in faces_of.items():
        by_file: dict = {}
        for face_key in face_keys:
            by_file.setdefault(face_key.split("/", 1)[0], []).append(face_key)
        if len(by_file) < min_files:
            continue
        name = state.identities.get(identity_id, {}).get("name") or ""
        events = set()
        file_rows = []
        for file_id in sorted(by_file):
            meta = file_meta.get(file_id, {})
            path = meta.get("path", "")
            event = posixpath.dirname(path) if path else "(unknown)"
            events.add(event)
            file_rows.append({
                "identity_id": identity_id,
                "name": name,
                "source_file_id": file_id,
                "file_path": path,
                "media_type": meta.get("media_type", ""),
                "n_faces_in_file": len(by_file[file_id]),
                "drive_link":
                    f"https://drive.google.com/file/d/{file_id}/view",
            })
        people.append({
            "identity_id": identity_id,
            "name": name,
            "n_faces": len(face_keys),
            "n_files": len(by_file),
            "n_events": len(events),
            "example_events": " | ".join(sorted(events)[:3]),
        })
        person_files.extend(file_rows)
        sample_faces[identity_id] = [
            by_file[file_id][0] for file_id
            in sorted(by_file)[:GALLERY_THUMBS_PER_PERSON]]

    people.sort(key=lambda p: (-p["n_files"], p["identity_id"]))
    rank = {p["identity_id"]: i for i, p in enumerate(people)}
    person_files.sort(key=lambda r: (rank[r["identity_id"]], r["file_path"]))
    return people, person_files, sample_faces


_GALLERY_STYLE = """
body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa;
       color: #222; max-width: 1100px; }
h1 { font-size: 22px; } .sub { color: #555; margin-bottom: 20px; }
.person { background: #fff; border: 1px solid #ddd; border-radius: 10px;
          padding: 12px 16px; margin: 10px 0; display: flex; gap: 16px;
          align-items: center; }
.person img { width: 96px; height: 96px; object-fit: cover;
              border-radius: 6px; margin-right: 4px; }
.person .who { min-width: 260px; }
.person .who b { font-size: 16px; }
.person .id { color: #888; font-size: 12px; font-family: monospace; }
.person .ev { color: #666; font-size: 12px; margin-top: 4px; }
.unnamed b { color: #a33; }
"""


def render_gallery(people: list, sample_faces: dict, thumbs: dict, *,
                   title: str, generated_at: str, note: str = "") -> str:
    """Self-contained HTML: thumbs maps face_key -> data URI (or None)."""
    cards = []
    for person in people:
        identity_id = person["identity_id"]
        images = "".join(
            f"<img src='{uri}' alt=''>"
            for uri in (thumbs.get(face_key)
                        for face_key in sample_faces.get(identity_id, []))
            if uri)
        display = person["name"] or "Who is this?"
        cards.append(
            f"<div class='person{'' if person['name'] else ' unnamed'}'>"
            f"<div>{images}</div>"
            f"<div class='who'><b>{html.escape(display)}</b><br>"
            f"<span class='id'>{html.escape(identity_id)}</span><br>"
            f"{person['n_files']} files · {person['n_events']} events · "
            f"{person['n_faces']} faces"
            f"<div class='ev'>{html.escape(person['example_events'])}</div>"
            f"</div></div>")
    n_unnamed = sum(1 for p in people if not p["name"])
    body = (
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='sub'>{len(people)} people · generated "
        f"{html.escape(generated_at)} · {n_unnamed} still unnamed — for any "
        f"face you recognize, note the <code>identity_...</code> code and "
        f"the name.{(' ' + html.escape(note)) if note else ''}</div>"
        + "".join(cards))
    return (f"<!doctype html><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            f"<style>{_GALLERY_STYLE}</style>{body}")
