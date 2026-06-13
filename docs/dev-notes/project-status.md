# Project status — slice 5 complete

> **ACTIVE WORK-STREAM (2026-06-13): FOLDER 2 detect-only bulk run.**
> A separate, long-running job (auto-advancing 33 × 3 TB chunks, ~18 days) is
> in progress — independent of the slice 1–5 share pipeline documented below.
> For its current state and the stop/resume/monitor runbook, see the
> "⮕ RESUME HERE" block at the top of
> `dev-log/2026-06-11-detect-only-drive-run.md` (dev-log is gitignored / local).

## Current state

- Slices 1-4 shipped to main. Pipeline auto-generates share package,
  produces single-face crops per cluster, uploads them to Drive, writes
  Drive URLs into face_clusters_summary.csv's contact_sheet column, and
  auto-uploads face_clusters_summary.csv and face_clusters_share.csv to
  the configured Drive folder.
- Validated end-to-end at scale: 180 files → 5888 faces → 530 clusters →
  530 Drive uploads → 530 IMAGE() formulas rendering in spreadsheet.
- Slice 4 shipped. CSV uploads update in place by name so file IDs stay
  stable across runs. face_clusters.csv (raw, with local paths) stays
  local. Pipeline exits non-zero on CSV upload failure so automation can
  detect broken runs.
- Slice 4 validated end-to-end against real Drive: ID stability,
  mimeType correctness, duplicate-name error handling, and exit code
  propagation all pass.
- Slice 5 shipped: end-user setup guide at docs/SETUP_FOR_USERS.md
  covering Mac and Windows from empty laptop to first run, with Google
  OAuth walkthrough (including the unverified-app warning bypass) and
  known footguns inlined. Main README points to it for non-developer
  audiences. Existing docs/SETUP.md flagged as outdated with a banner.
- Cross-platform install validated on Windows: pipeline runs end-to-end
  after pip-based install path against the api-test folder. tf-keras
  was missing on the Windows pip path; added to requirements.txt.
- Spreadsheet (Shivang's UI) lives in slice3_dev_target/ in team Drive
  (moved out of Converge's 10GB folder due to recursive-scan pickup).
- Team's Apps Script script updated for slice 3:
  syncFaceLibraryFromSummaryCSV reads URLs from CSV directly (no more
  searchFiles lookup). updateFileDescriptions regex updated to handle
  uc?export=view&id= URL format. fillDriveLinks and importFromCSV still
  have team-Drive folder IDs hardcoded — not blocking for the usability
  test but need updating before broader Converge use.

## Architecture decisions worth remembering

- Pipeline-first design: pipeline outputs are self-describing, scripts
  adapt to consume cleaner outputs rather than pipeline accommodating
  script conventions.
- Caches are folder-keyed by Drive ID: outputs/pipeline/downloads/
  (~11GB after a full run) and outputs/pipeline/processed/ (embedding
  JSONs) survive across runs. Re-running with caches intact skips
  download and embedding entirely; only clustering and uploads happen.
- Slice 3 destination folder must be OUTSIDE the source media tree, or
  recursive scan picks up the pipeline's own outputs as input media.
- requirements.txt does not pin tensorflow explicitly; DeepFace pulls
  it transitively. Worth pinning later if a Windows install ever
  surfaces a version conflict.
- environment.yml is macOS-pinned (tensorflow-macos, dbus, etc.).
  Windows install path is conda+pip rather than env-from-file.
  Documented in SETUP_FOR_USERS.md but no separate environment-windows.yml
  yet.

## What's outstanding

Post-usability-test candidates, roughly in priority order:

- Stale-cache invalidation. If Converge edits source files in place on
  Drive, the pipeline's cache uses stale embeddings. Footgun worth
  asking them about; may not be a real issue in their workflow.
- Apps Script folder-ID cleanup for Converge Drive context.
  fillDriveLinks and importFromCSV still hardcode team-Drive IDs.
- Real cross-platform install story. Either a separate
  environment-windows.yml or moving away from conda-pinned approach
  toward requirements.txt + smaller conda base.
- Begin feedback-loop work per docs/future-work/feedback-loop.md. The
  largest planned workstream; should be informed by usability-test
  observations rather than designed in advance.

## Reference

- docs/future-work/feedback-loop.md — design notes for the
  human-in-the-loop identity labeling system. Largest planned
  post-user-test workstream.
- docs/SETUP_FOR_USERS.md — end-user setup guide.
- README.md — developer-audience documentation.
