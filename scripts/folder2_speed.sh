#!/usr/bin/env bash
# Switch the FOLDER 2 run between FULL speed and the SCHEDULED throttle.
#
#   scripts/folder2_speed.sh full        # no cap, ever
#   scripts/folder2_speed.sh scheduled   # 60 MB/s outside 02:00-08:00 (default)
#   scripts/folder2_speed.sh status      # show current mode (no restart)
#
# Mechanism: writes BW_CAP/BW_WINDOW to the systemd EnvironmentFile and restarts
# the service. Restart resumes per-file via progress.jsonl (only the <=8
# in-flight downloads re-fetch, ~15s TF warmup), so toggling is cheap. The mode
# persists across reboots. In SCHEDULED mode the 02:00/08:00 flips happen on
# their own inside the running process — you only need this to override that.
set -u

ENVFILE="$HOME/.config/folder2-detect.env"
UNIT=folder2-detect
CAP=60          # MB/s in scheduled mode
WINDOW=2-8      # full-speed local-hour window in scheduled mode

now_mode() {
  local cap="$CAP"
  [ -f "$ENVFILE" ] && cap=$(grep -E '^BW_CAP=' "$ENVFILE" | tail -1 | cut -d= -f2)
  [ "${cap:-$CAP}" = "0" ] && echo "FULL" || echo "SCHEDULED"
}

case "${1:-status}" in
  full)
    mkdir -p "$(dirname "$ENVFILE")"
    printf 'BW_CAP=0\nBW_WINDOW=%s\n' "$WINDOW" > "$ENVFILE"
    systemctl --user restart "$UNIT"
    echo "-> FULL SPEED (no cap). service restarted; resumes from progress.jsonl."
    ;;
  scheduled)
    mkdir -p "$(dirname "$ENVFILE")"
    printf 'BW_CAP=%s\nBW_WINDOW=%s\n' "$CAP" "$WINDOW" > "$ENVFILE"
    systemctl --user restart "$UNIT"
    echo "-> SCHEDULED (${CAP} MB/s outside ${WINDOW//-/:00-}:00). service restarted."
    ;;
  status)
    echo "mode    : $(now_mode)"
    echo "envfile : $ENVFILE"
    if [ -f "$ENVFILE" ]; then sed 's/^/          /' "$ENVFILE"; else echo "          (none -> driver defaults BW_CAP=${CAP} BW_WINDOW=${WINDOW})"; fi
    echo "service : $(systemctl --user is-active "$UNIT" 2>/dev/null)"
    h=$(date +%H)
    if [ "$(now_mode)" = "FULL" ]; then
      echo "now     : uncapped (full speed)"
    elif [ "$h" -ge "${WINDOW%-*}" ] && [ "$h" -lt "${WINDOW#*-}" ]; then
      echo "now     : full-speed window (${WINDOW//-/:00-}:00) -> uncapped"
    else
      echo "now     : throttle window -> capped to ${CAP} MB/s"
    fi
    ;;
  *)
    echo "usage: $0 {full|scheduled|status}" >&2
    exit 2
    ;;
esac
