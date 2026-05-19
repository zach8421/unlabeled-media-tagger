# CLAUDE.md — unlabeled-media-tagger

Durable context for agents working in this project. Specific tasks belong
in the prompt; what changes infrequently belongs here.

## Python environment

Use the project venv: `./.venv/bin/python`. It has `cv2`, `deepface`,
`google-api-python-client`, `numpy`, and the rest of the runtime
dependencies. Tests run via `./.venv/bin/python -m pytest tests/`.

The `datasci` conda env (referenced in the parent CLAUDE.md for grad-class
work) does **not** have `cv2` or `deepface` — don't reach for it inside this
project. The notebook generator
[scripts/build_face_preprocessing_notebook.py](scripts/build_face_preprocessing_notebook.py)
only needs `nbformat`, which the parent `datasci` env happens to have.

## Pipeline architecture

There are two execution paths against the same logical pipeline:

- **Local CLI pipeline** — `src/unlabeled_media_tagger/pipeline/run.py`,
  entry via [scripts/run_local_pipeline.py](scripts/run_local_pipeline.py).
  Full end-to-end: Drive fetch → frame extract → detect → embed → cluster
  → share-package → Drive upload. Developer-driven, OAuth via secrets/.

- **Colab notebook** — [examples/face_preprocessing_colab.ipynb](examples/face_preprocessing_colab.ipynb).
  Sponsor-facing, currently preprocessing-only: detect + crop → per-folder
  `asset_faces.csv` + unified `preprocessing_manifest.csv`. The intent
  (per dev-log) is to grow it end-to-end so sponsors don't need a CLI
  install. **The notebook is generated** from
  [scripts/build_face_preprocessing_notebook.py](scripts/build_face_preprocessing_notebook.py)
  — edit the generator's cell-source string constants, not the .ipynb
  JSON. Regenerate via
  `python scripts/build_face_preprocessing_notebook.py examples/face_preprocessing_colab.ipynb`.

The notebook must stay **self-contained**: no
`from unlabeled_media_tagger...` imports inside notebook cells. Helpers
that live in `src/unlabeled_media_tagger/` are copy-pasted into the
notebook's helpers cell verbatim by the generator. The lib version is
the source of truth and gets tested; the notebook copy is for
distribution.

## Development log

Working notes that are too granular for git commits live in `dev-log/`.
Before starting non-trivial work, scan the most recent 1–3 entries (and
anything matching your topic by slug) for context. After completing a
logical work-stream, write a new entry following
[dev-log/README.md](dev-log/README.md).

The dev-log is **gitignored**: it's a local journal, not a shipping
artifact.

## Current state

[docs/dev-notes/project-status.md](docs/dev-notes/project-status.md) is
the latest single-snapshot summary of where slices 1–5 landed and what's
outstanding. Read that first if joining the project cold.

## Sponsor context (Converge Media)

- **Sponsor is non-technical.** CLI is a non-starter for them. The Colab
  notebook is the sponsor-facing deliverable; the local pipeline is for
  development and the validation runs that test against real Drive data.
  Don't propose plans that push technical steps back onto the sponsor.
- **One Drive folder = one event = one shoot.** Useful constraint when
  reasoning about moments / clothing consistency / per-event tuning.
