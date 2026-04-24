#!/bin/bash
# Runs in tmux: syncs results, rebuilds dashboard, deploys to gh-pages every INTERVAL minutes.

set -euo pipefail

INTERVAL=${1:-30}  # minutes, default 30
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/work/pi_nwycoff_umass_edu/.conda/envs/hang/bin/python"
export PATH="$HOME/.local/bin:$PATH"

echo "[auto_deploy] Starting — root=$ROOT, interval=${INTERVAL}m"
echo "[auto_deploy] Press Ctrl-C to stop."
echo ""

run_once() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running update-all..."
    cd "$ROOT"
    if $PYTHON scripts/cli.py update-all --allow-dirty 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Next run in ${INTERVAL}m."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] update-all failed (see above). Will retry in ${INTERVAL}m."
    fi
    echo ""
}

run_once
while true; do
    sleep $((INTERVAL * 60))
    run_once
done
