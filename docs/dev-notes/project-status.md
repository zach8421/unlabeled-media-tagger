# Project status — slice 4 complete

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
  propagation all pass. No CSV upload still required for the user test.
- Spreadsheet (Shivang's UI) lives in slice3_dev_target/ in team Drive
  (moved out of Converge's 10GB folder due to recursive-scan pickup).
- Team's Apps Script script updated for slice 3: syncFaceLibraryFromSummaryCSV
  reads URLs from CSV directly (no more searchFiles lookup).
  updateFileDescriptions regex updated to handle uc?export=view&id= URL format.
  fillDriveLinks and importFromCSV still have your-team-Drive folder IDs
  hardcoded — not blocking but need updating before user test.

## Architecture decisions worth remembering

- Pipeline-first design: pipeline outputs are self-describing, scripts adapt
  to consume cleaner outputs rather than pipeline accommodating script
  conventions.
- Caches are folder-keyed by Drive ID: outputs/pipeline/downloads/ (11GB)
  and outputs/pipeline/processed/ (embedding JSONs) survive across runs.
- Slice 3 destination folder must be OUTSIDE the source media tree, or
  recursive scan picks up the pipeline's own outputs as input media.

## What's outstanding

[list of slice candidates, see below]

## Reference

- docs/future-work/feedback-loop.md — design notes for the human-in-the-loop
  identity labeling system. Largest planned post-user-test workstream.
