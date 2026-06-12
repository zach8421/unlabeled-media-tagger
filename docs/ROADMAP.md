# Roadmap

Canonical "what's next" for unlabeled-media-tagger. Routine progress and
granular reasoning live in `dev-log/`; this file tracks major decisions
and the active focus.

**Last updated:** 2026-05-18

## Current focus

**Self-contained Colab notebook**, target horizon ~2 weeks. The sponsor
(Converge Media) is non-technical, so the CLI-based local pipeline is
not a viable delivery vehicle for them. The notebook is being grown
from preprocessing-only to end-to-end so they can run the full pipeline
without a local install.

The local pipeline is in maintenance — see the second track below.

## Notebook track — active

**Current state.** v1.1 shipped (see
[dev-log/2026-05-18-v1.1-blur-filter.md](../dev-log/2026-05-18-v1.1-blur-filter.md)).
The notebook covers Drive folder → face detection → per-face crops with a
v1.1 quality gate (`quality_score` column plus `filtered_blurry` /
`filtered_small_crop` statuses) → per-folder `asset_faces.csv` → unified
`preprocessing_manifest.csv`. Does **not** yet do embeddings, clustering,
contact sheets, or spreadsheet upload.

### Next major lift: end-to-end in the notebook

Goal: a sponsor pastes a Drive folder URL into the configuration cell,
runs all cells, and gets a populated spreadsheet at the end. Largest
single piece of remaining work. The local pipeline already does all of
this — the lift is porting the stages into the notebook while keeping
it self-contained (no `from unlabeled_media_tagger...` imports).

Components to port:

- Face embedding (ArcFace via DeepFace) over the v1.1 crops.
- Clustering — start face-only with the existing `CompareStage` policy
  (online centroid, similarity threshold 0.68). Two-pass HAC is a
  queued enhancement below.
- Share package: a representative crop per cluster plus contact sheets.
- Drive upload of contact sheets and the summary CSV with stable file
  IDs (overwrite-by-name, per the slice-4 pattern in the local pipeline).
- Apps Script integration end-to-end — the existing
  `syncFaceLibraryFromSummaryCSV` consumer should work as-is if the
  notebook emits the same column shape as the local pipeline.

Open design questions (auth flow inside Colab, output-path conventions,
Colab compute budget) will be worked through in the dev-log entry for
that work-stream.

### Queued after end-to-end exists

Sequenced because each depends on a face-only baseline existing first
so the win is measurable. All three are described in more depth in the
v1.1 dev-log entry; sources are the Apple "Recognizing People in
Photos" paper unless noted.

1. **Body / upper-body embedding.** Combined distance
   `D = min(F, αF + βT)`, restricted to within-folder matching (one
   Drive folder = one event = one moment, clothing constant — Apple's
   constraint maps cleanly to Converge's shoots). Best win expected on
   profile / mic-occluded / back-turned shots common in debate
   photography. Adds a body detector, face↔body pairing, and a body
   embedding model. Start cheap with CLIP image embeddings on the upper
   body; upgrade to a person re-ID model if needed.

2. **Two-pass agglomerative clustering.** Greedy first pass within
   moment for tight, high-precision clusters; HAC second pass to grow
   them across moments. Drop-in replacement for the current
   online-centroid `CompareStage`. Possibly higher impact per unit of
   effort than body embedding alone.

3. **Multi-model face ensemble** — ArcFace + FaceNet + VGG-Face via
   DeepFace, distances combined at cluster time. Cheaper intermediate
   if (1) or (2) turn out to be heavier than expected; diversity in
   models often captures most of what diversity in features would.

### Open / minor

- **Blur/size threshold validation on other Converge shoots.** The
  current values (`BLUR_THRESHOLD=45.0`, `MIN_CROP_DIM=100`) were tuned
  on one debate dataset. Re-validate when other event types arrive
  (different lighting, different backdrops). Side tool:
  [scripts/blur_filter_tune.py](../scripts/blur_filter_tune.py).

## Local pipeline track — maintenance

**Status:** slices 1–5 shipped end-to-end against real Drive (see
[docs/dev-notes/project-status.md](dev-notes/project-status.md) for the
state-of-the-world snapshot). No active feature development planned;
bug fixes and small usability tweaks welcome.

### Outstanding items

Real but unscheduled. Will get attention if they block sponsor usage or
block notebook-track work.

- **Stale-cache invalidation.** If Converge edits source files in place
  on Drive, the embedding cache uses stale embeddings. Probably not a
  real issue in their workflow but worth verifying.
- **Apps Script folder-ID cleanup.** `fillDriveLinks` and
  `importFromCSV` still hardcode team-Drive IDs. Needs updating before
  broader Converge rollout.
- **Cross-platform install hardening.** Either a separate
  `environment-windows.yml` or a pip-based path with a smaller conda
  base. Currently soft in `docs/SETUP_FOR_USERS.md`.
- **Feedback-loop design** — see
  [docs/future-work/feedback-loop.md](future-work/feedback-loop.md).
  Largest single planned workstream. Likely gets adopted into the
  notebook track once the notebook reaches end-to-end and Converge has
  real usage data, rather than pursued separately on the local side.

## How to use this file

- **Canonical "what's next."** When asking "should I work on X?" the
  answer should be derivable from this file plus the relevant dev-log
  entry.
- **Dev-log entries link back here** in their `Related` field when
  they advance a roadmap item.
- **Updated when major focus shifts**, not on routine progress.
  Track-level changes (active → maintenance, new track opens, item
  promoted out of "queued") belong here. Tuning observations and
  per-work-stream decisions live in dev-log.
- **The `Last updated` date is load-bearing.** When dev-log entries
  reference this file, they're referencing the version as of that
  date. If you're reading this and the date is more than a month old,
  treat the queued items with appropriate skepticism — re-check
  against current sponsor needs and the most recent dev-log entries
  before committing to a sequence.
