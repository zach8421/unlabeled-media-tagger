# Project status — slice 3 complete

## Current state

- Slices 1-3 shipped to main. Pipeline auto-generates share package,
  produces single-face crops per cluster, uploads them to Drive, and
  writes Drive URLs into face_clusters_summary.csv's contact_sheet column.
- Validated end-to-end at scale: 180 files → 5888 faces → 530 clusters →
  530 Drive uploads → 530 IMAGE() formulas rendering in spreadsheet.
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
