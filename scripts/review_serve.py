#!/usr/bin/env python
"""Start the local review server for one cluster run.

  PYTHONPATH=src ./.venv/bin/python scripts/review_serve.py \
      --run-dir /mnt/media1/folder2_cluster/run_001 \
      [--labels-dir /mnt/media1/folder2_labels] [--port 8765]

Then browse http://127.0.0.1:8765/ — every mark persists to the label store
immediately. Ctrl-C to stop; restart any time, nothing is lost.
"""

from __future__ import annotations

import argparse
import sys

from unlabeled_media_tagger.review.server import serve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--labels-dir", default="/mnt/media1/folder2_labels")
    ap.add_argument("--crops-root", default="/mnt/media1/folder2_detect/crops")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)

    server = serve(args.labels_dir, args.run_dir,
                   crops_root=args.crops_root, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
