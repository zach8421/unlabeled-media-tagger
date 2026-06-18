#!/usr/bin/env bash
# Read-only progress summary for the FOLDER 2 detect-only run.
# Usage: scripts/folder2_status.sh [--rate]   (--rate adds a 10s throughput sample)
set -u
OUT=/mnt/media1/folder2_detect
MANIFEST=outputs/manifests/folder2_clean.csv
PROG="$OUT/progress.jsonl"

# total target files (clean manifest minus header)
total=$(($(wc -l < "$MANIFEST" 2>/dev/null || echo 1) - 1))
# grep -c already prints "0" on no match (just with exit 1); use `|| true` to
# swallow the exit code WITHOUT echoing a second "0".
recorded=$(wc -l < "$PROG" 2>/dev/null || echo 0)
ok=$(grep -c '"status": "ok"' "$PROG" 2>/dev/null || true)
nofaces=$(grep -c '"status": "no_faces"' "$PROG" 2>/dev/null || true)
unread=$(grep -c '"status": "unreadable"' "$PROG" 2>/dev/null || true)
derr=$(grep -c '"status": "detect_error"' "$PROG" 2>/dev/null || true)
dlerr=$(grep -c '"status": "download_error"' "$PROG" 2>/dev/null || true)
pct=$(awk -v a="$recorded" -v b="$total" 'BEGIN{printf "%.1f", (b>0? 100*a/b : 0)}')

echo "service : $(systemctl --user is-active folder2-detect 2>/dev/null) ($(systemctl --user is-enabled folder2-detect 2>/dev/null))"
echo "progress: $recorded / $total files (${pct}%)"
echo "  ok=$ok  no_faces=$nofaces  unreadable=$unread  detect_err=$derr  download_err=$dlerr"
echo "chunk   : $(grep -E 'chunk .* (START|DONE)' "$OUT/driver.log" 2>/dev/null | tail -1 | sed 's/.*| //')"
echo "crops   : $(find "$OUT/crops" -type f 2>/dev/null | wc -l) face crops on disk"
echo "media1  : $(df -h /mnt/media1 2>/dev/null | awk '/mnt/{print $4" free"}')"

if [ "${1:-}" = "--rate" ]; then
  iface=$(ip route show default | awk '{print $5; exit}')
  r1=$(cat /sys/class/net/$iface/statistics/rx_bytes); sleep 10
  r2=$(cat /sys/class/net/$iface/statistics/rx_bytes)
  awk -v a=$r1 -v b=$r2 'BEGIN{m=(b-a)/1e6/10; printf "rate    : %.1f MB/s (%.0f Mbps) on '"$iface"'\n", m, m*8}'
fi
