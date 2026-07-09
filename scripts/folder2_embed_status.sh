#!/usr/bin/env bash
# Read-only progress summary for the FOLDER 2 embedding run.
# Usage: scripts/folder2_embed_status.sh
set -u
OUT=/mnt/media1/folder2_embed
PROG="$OUT/progress.jsonl"
MANIFEST="$OUT/embed_manifest.csv"

total_faces=$(($(wc -l < "$MANIFEST" 2>/dev/null || echo 1) - 1))
# 8192 faces per shard (run_embed_batch.py default).
total_shards=$(( (total_faces + 8191) / 8192 ))

done_shards=0; faces=0; failed=0; last_rate="-"
if [ -f "$PROG" ]; then
  done_shards=$(grep -c '"status": "ok"' "$PROG" || true)
  faces=$(awk -F'"faces": ' '/"status": "ok"/{split($2,a,","); s+=a[1]} END{print s+0}' "$PROG")
  failed=$(awk -F'"read_failed": ' '/"status": "ok"/{split($2,a,","); s+=a[1]} END{print s+0}' "$PROG")
  last_rate=$(tail -1 "$PROG" | grep -o '"faces_per_sec": [0-9.]*' | cut -d' ' -f2)
fi

pct=$(awk -v a="$done_shards" -v b="$total_shards" 'BEGIN{printf "%.1f", (b>0? 100*a/b : 0)}')
echo "service : $(systemctl --user is-active folder2-embed 2>/dev/null)"
echo "shards  : $done_shards / $total_shards (${pct}%)"
echo "faces   : $faces embedded, $failed read-failed (target $total_faces)"
echo "rate    : ${last_rate:-–} faces/s (last shard)"
echo "output  : $(ls "$OUT/embeddings" 2>/dev/null | wc -l) npy shards, $(du -sh "$OUT/embeddings" 2>/dev/null | cut -f1) on disk"
echo "media1  : $(df -h /mnt/media1 2>/dev/null | awk '/mnt/{print $4" free"}')"
if [ -f "$OUT/faces_index.csv" ]; then
  echo "finalize: faces_index.csv present ($(($(wc -l < "$OUT/faces_index.csv") - 1)) rows)"
fi
