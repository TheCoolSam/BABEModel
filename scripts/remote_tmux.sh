#!/usr/bin/env bash
# Run strength-upgrade experiments in a detached tmux session (SSH-friendly).
# Usage: bash scripts/remote_tmux.sh [iters=20] [workers=4]
set -euo pipefail
ITERS="${1:-20}"
WORKERS="${2:-4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SESSION="babe_experiments"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "python -m experiments.ablations --iters=$ITERS --workers=$WORKERS; \
   python -m experiments.baselines --iters=$ITERS --workers=$WORKERS; \
   python -m experiments.topology_robustness --iters=15 --workers=$WORKERS; \
   echo DONE; bash"

echo "Started tmux session '$SESSION'."
echo "  attach: tmux attach -t $SESSION"
echo "  Closing your laptop is fine; the remote host keeps running."
