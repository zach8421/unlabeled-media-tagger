#!/usr/bin/env python3
"""Score a folder of face crops and produce a tuning report.

Walks `--crops-dir` recursively, scores every JPEG/PNG with
laplacian_variance, and writes:

  <out>/scores.csv   one row per crop, sorted by score ascending
  <out>/report.html  contact sheet sorted by score, with the threshold line
                     and a highlighted borderline window for visual tuning

The borderline window (--window) is the band of crops whose score sits within
N units on either side of the threshold. Those are the crops to inspect first
when picking a threshold: are the ones below the line actually unusable, and
are the ones above the line actually fine?

If `--asset-faces-csv` is provided, source-file lineage from the notebook's
asset_faces.csv is joined onto each crop via the crop_file_name column. This
lets the report show "which source image did this borderline crop come from."

Typical workflow:
    1. Run with a guess threshold (start at 100 for Laplacian variance).
    2. Open report.html in a browser, scroll the borderline band.
    3. Adjust the threshold up or down until the line sits where your eye
       agrees the crops cross from usable to unusable. Re-run.
"""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

# scripts/ isn't on sys.path by default; the package lives under src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from unlabeled_media_tagger.preprocessing.blur import score_crop  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--crops-dir", required=True, type=Path,
                   help="Folder of face crops. Walked recursively.")
    p.add_argument("--threshold", required=True, type=float,
                   help="Laplacian-variance score below which a crop would be "
                        "filtered. Start around 100 and tune.")
    p.add_argument("--window", type=float, default=20.0,
                   help="Half-width of the borderline band around the "
                        "threshold (in score units). Default 20.")
    p.add_argument("--min-crop-dim", type=int, default=0,
                   help="Quality-stage minimum: crops where the short side is "
                        "below this get band 'filtered_small_crop' regardless "
                        "of score. Defends against high-frequency-but-tiny "
                        "crops (halftone backdrops, etc.). Default 0 = off.")
    p.add_argument("--asset-faces-csv", type=Path, default=None,
                   help="Optional asset_faces.csv from the Colab notebook. "
                        "When provided, source-file lineage is joined onto "
                        "each crop.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for scores.csv and report.html.")
    return p.parse_args()


def find_crops(crops_dir: Path) -> list[Path]:
    return sorted(
        p for p in crops_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_lineage(csv_path: Path) -> dict[str, dict]:
    """Map crop_file_name -> {source_file_name, face_id, confidence, bbox}."""
    lineage: dict[str, dict] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("crop_file_name") or ""
            if not name:
                continue
            lineage[name] = {
                "source_file_name": row.get("source_file_name", ""),
                "face_id": row.get("face_id", ""),
                "confidence": row.get("confidence", ""),
                "bbox_w": row.get("bbox_w", ""),
                "bbox_h": row.get("bbox_h", ""),
            }
    return lineage


def classify(score: float | None, threshold: float, window: float,
             width: int | None = None, height: int | None = None,
             min_crop_dim: int = 0) -> str:
    if score is None:
        return "unreadable"
    if min_crop_dim > 0 and width is not None and height is not None:
        if min(width, height) < min_crop_dim:
            return "filtered_small_crop"
    if score < threshold - window:
        return "would_filter"
    if score < threshold:
        return "borderline_below"
    if score < threshold + window:
        return "borderline_above"
    return "pass"


def write_scores_csv(rows: list[dict], out_path: Path) -> None:
    cols = ["crop_path", "crop_file_name", "width", "height", "score",
            "band", "source_file_name", "face_id", "confidence",
            "bbox_w", "bbox_h", "error"]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


BAND_STYLE = {
    "would_filter": ("#f8d7da", "would filter (blurry)"),
    "filtered_small_crop": ("#f5c6cb", "filtered (small crop)"),
    "borderline_below": ("#fff3cd", "borderline (below)"),
    "borderline_above": ("#d1ecf1", "borderline (above)"),
    "pass": ("#d4edda", "pass"),
    "unreadable": ("#e2e3e5", "unreadable"),
}


def write_html_report(rows: list[dict], out_path: Path, threshold: float,
                      window: float, crops_dir: Path) -> None:
    counts = {k: 0 for k in BAND_STYLE}
    for r in rows:
        counts[r["band"]] = counts.get(r["band"], 0) + 1

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>blur tuning report</title>")
    parts.append("<style>")
    parts.append("body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;}")
    parts.append("h1{margin-bottom:8px;} .meta{color:#555;margin-bottom:20px;}")
    parts.append(".legend span{display:inline-block;padding:4px 10px;margin-right:8px;border-radius:4px;font-size:13px;}")
    parts.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-top:20px;}")
    parts.append(".card{padding:6px;border-radius:6px;font-size:12px;line-height:1.3;}")
    parts.append(".card img{max-width:100%;height:auto;display:block;border-radius:3px;}")
    parts.append(".card .score{font-weight:600;font-size:14px;}")
    parts.append(".card .meta2{color:#444;}")
    parts.append(".threshold-bar{margin:24px 0 4px;border-top:3px dashed #c00;padding-top:6px;color:#c00;font-weight:600;}")
    parts.append("</style></head><body>")
    parts.append(f"<h1>blur tuning report</h1>")
    parts.append(f"<div class='meta'>crops_dir: <code>{html.escape(str(crops_dir))}</code><br>")
    parts.append(f"threshold = <b>{threshold:g}</b> &nbsp; window = ±{window:g} &nbsp; n = {len(rows)}</div>")

    parts.append("<div class='legend'>")
    for band, (bg, label) in BAND_STYLE.items():
        parts.append(f"<span style='background:{bg}'>{label}: {counts.get(band, 0)}</span>")
    parts.append("</div>")

    # Sorted ascending; insert a threshold bar at the boundary.
    parts.append("<div class='grid'>")
    threshold_inserted = False
    for r in rows:
        if not threshold_inserted and r["score"] is not None and r["score"] >= threshold:
            parts.append(f"</div><div class='threshold-bar'>threshold = {threshold:g}</div><div class='grid'>")
            threshold_inserted = True
        bg, _ = BAND_STYLE[r["band"]]
        score_text = f"{r['score']:.1f}" if r["score"] is not None else "(unreadable)"
        src_label = r.get("source_file_name") or Path(r["crop_path"]).name
        parts.append(f"<div class='card' style='background:{bg}'>")
        parts.append(f"<img src='file://{html.escape(str(Path(r['crop_path']).resolve()))}' alt=''>")
        parts.append(f"<div class='score'>{score_text}</div>")
        parts.append(f"<div class='meta2'>{html.escape(Path(r['crop_path']).name)}</div>")
        if r.get("source_file_name") and r["source_file_name"] != src_label:
            parts.append(f"<div class='meta2'>from: {html.escape(r['source_file_name'])}</div>")
        if r.get("bbox_w") and r.get("bbox_h"):
            parts.append(f"<div class='meta2'>bbox: {html.escape(r['bbox_w'])}×{html.escape(r['bbox_h'])}</div>")
        parts.append("</div>")
    parts.append("</div>")
    parts.append("</body></html>")
    out_path.write_text("".join(parts))


def main() -> int:
    args = parse_args()
    crops_dir: Path = args.crops_dir.resolve()
    if not crops_dir.exists():
        print(f"ERROR: crops dir not found: {crops_dir}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    lineage = load_lineage(args.asset_faces_csv) if args.asset_faces_csv else {}

    crops = find_crops(crops_dir)
    print(f"Found {len(crops)} crop(s) under {crops_dir}")

    rows: list[dict] = []
    for path in crops:
        scored = score_crop(path)
        crop_file_name = path.name
        lin = lineage.get(crop_file_name, {})
        rows.append({
            "crop_path": scored["path"],
            "crop_file_name": crop_file_name,
            "width": scored["width"] if scored["width"] is not None else "",
            "height": scored["height"] if scored["height"] is not None else "",
            "score": scored["score"],
            "band": classify(
                scored["score"], args.threshold, args.window,
                width=scored["width"], height=scored["height"],
                min_crop_dim=args.min_crop_dim,
            ),
            "error": scored.get("error") or "",
            **lin,
        })

    # Sort ascending; unreadable rows (score=None) sink to the bottom.
    rows.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))

    scores_csv = args.out / "scores.csv"
    report_html = args.out / "report.html"
    # Stringify score for CSV after sorting on the float.
    csv_rows = [{**r, "score": ("" if r["score"] is None else f"{r['score']:.4f}")} for r in rows]
    write_scores_csv(csv_rows, scores_csv)
    write_html_report(rows, report_html, args.threshold, args.window, crops_dir)

    counts = {k: 0 for k in BAND_STYLE}
    for r in rows:
        counts[r["band"]] += 1
    print(f"Scored {len(rows)} crop(s).")
    print(f"Threshold = {args.threshold:g}  Window = ±{args.window:g}  "
          f"min_crop_dim = {args.min_crop_dim}")
    if counts["filtered_small_crop"]:
        print(f"  filtered_small_crop (min(w,h) < {args.min_crop_dim}): {counts['filtered_small_crop']}")
    print(f"  would_filter    (score < {args.threshold - args.window:g}): {counts['would_filter']}")
    print(f"  borderline_below ({args.threshold - args.window:g} .. {args.threshold:g}): {counts['borderline_below']}")
    print(f"  borderline_above ({args.threshold:g} .. {args.threshold + args.window:g}): {counts['borderline_above']}")
    print(f"  pass            (score >= {args.threshold + args.window:g}): {counts['pass']}")
    if counts["unreadable"]:
        print(f"  unreadable: {counts['unreadable']}")
    print(f"Wrote: {scores_csv}")
    print(f"Wrote: {report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
