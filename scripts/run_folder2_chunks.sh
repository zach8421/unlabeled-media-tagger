#!/usr/bin/env bash
# Auto-advance driver for the FOLDER 2 detect-only run.
#
# Runs the 3 TB chunk manifests (outputs/manifests/folder2_chunks/chunk_NNN.csv)
# sequentially into ONE unified output dir. Detection is single-GPU; downloads
# are 8-wide via run_detect_only.py's pool.
#
# SAFE TO STOP AND RE-RUN AT ANY TIME. Resume is per-file via the unified
# progress.jsonl, so:
#   - an interrupted chunk continues where it left off (only the <=8 in-flight
#     files at kill time re-download), and
#   - already-finished chunks no-op in seconds (to-process: 0).
# The driver is stateless: it just loops chunks; idempotency comes from
# progress.jsonl. Re-running after downtime resumes correctly.
#
#   Stop:   pkill -f run_detect_only.py   # current chunk dies; driver then exits
#   Resume: re-run this script with the same args
#
# Usage: scripts/run_folder2_chunks.sh [START] [END]   (defaults 1 32; 000 done)

set -u
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
MANIFEST_DIR=outputs/manifests/folder2_chunks
OUT=/mnt/media1/folder2_detect
SCRATCH=/mnt/ssd-scratch/detect-scratch/folder2
START=${1:-1}
END=${2:-32}

mkdir -p "$OUT" "$SCRATCH"
DRIVER_LOG="$OUT/driver.log"
log(){ echo "$(date -Is) | $*" | tee -a "$DRIVER_LOG"; }

log "driver start: chunks $START..$END  out=$OUT  scratch=$SCRATCH"
for i in $(seq "$START" "$END"); do
  n=$(printf "%03d" "$i")
  chunk="$MANIFEST_DIR/chunk_$n.csv"
  if [ ! -f "$chunk" ]; then log "MISSING $chunk, skipping"; continue; fi

  # Clear orphaned scratch left by any prior hard kill. Nothing here is needed:
  # unfinished files just re-download. Guarded so an unset/odd path can't nuke /.
  case "$SCRATCH" in
    /mnt/ssd-scratch/*) rm -rf "${SCRATCH:?}/"* 2>/dev/null ;;
    *) log "refusing to clear unexpected scratch path: $SCRATCH"; exit 2 ;;
  esac

  log "=== chunk $n START ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_detect_only.py "$chunk" \
      --output-dir "$OUT" --scratch-dir "$SCRATCH" \
      --frame-interval 5.0 --max-frames 60 --workers 8 --max-inflight-gb 200 \
      >> "$OUT/chunk_$n.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    # Non-zero = killed (you stopping it) or a real crash (disk full, etc).
    # Either way, stop the driver rather than burn through chunks while broken.
    log "chunk $n exited rc=$rc (killed or crashed) -> stopping driver"
    exit "$rc"
  fi
  log "=== chunk $n DONE ==="
done
log "driver finished all chunks $START..$END"
