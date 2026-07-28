#!/bin/bash
# Watch docs/our_paper_tex/**/*.tex (and .bib) for saves and auto-run build.sh.
# This emulates "Ctrl+S auto-compile" (like LaTeX Workshop's onSave build)
# without needing any editor extension, using inotify on the DolphinFS mount.
#
# Usage:
#   bash watch_build.sh            # run in foreground (Ctrl+C to stop)
#   nohup bash watch_build.sh > /tmp/watch_build.log 2>&1 &   # run in background
#
# Requires: inotify-tools (sudo yum install -y inotify-tools)

set -u
REPO_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/yangwenkui03/zwerge/code"
TEX_DIR="$REPO_ROOT/docs/our_paper_tex"
BUILD_SCRIPT="$REPO_ROOT/build.sh"
DEBOUNCE_SECONDS=1

if ! command -v inotifywait >/dev/null 2>&1; then
  echo "inotifywait not found. Install it with: sudo yum install -y inotify-tools" >&2
  exit 1
fi

echo "=== Watching for .tex/.bib/.sty saves under: $TEX_DIR ==="
echo "=== Will auto-run: bash $BUILD_SCRIPT ==="

last_build_ts=0

# -r: recursive, -m: monitor (keep running), close_write: fires once when the
# editor finishes writing+closing the file (i.e. on save).
# NOTE: this build of inotify-tools (3.14) does not support --include, so we
# filter by extension manually in the loop below.
inotifywait -m -r -e close_write --format '%w%f' \
  "$TEX_DIR" 2>/tmp/watch_build_inotify.log |
while read -r changed_file; do
  case "$changed_file" in
    *.tex|*.bib|*.sty|*.cls) ;;
    *) continue ;;
  esac

  now=$(date +%s)
  # Simple debounce: ignore triggers within DEBOUNCE_SECONDS of the last build
  # (editors sometimes emit multiple close_write events per save).
  if (( now - last_build_ts < DEBOUNCE_SECONDS )); then
    continue
  fi
  last_build_ts=$now
  echo ""
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Detected save: $changed_file ==="
  echo "=== Rebuilding... ==="
  if bash "$BUILD_SCRIPT"; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Build OK ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Build FAILED (see output above / AnonymousSubmission2027.log) ==="
  fi
done
