#!/bin/bash
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --requeue

# Runs one training stage. Env vars:
#   TRAIN_PY  — path to the design's train.py wrapper
#   STAGE     — "1" or "2"
#
# At the end of a successful stage-1 run, this script submits stage 2 only if
# either (a) the design is `baseline` (always proceed) or (b) the design's
# stage-1 composite_val beats baseline's stage-1 composite_val.

set -u

ROOT_DIR=${ROOT_DIR:-/work/pi_nwycoff_umass_edu/hang/autosapiens_iter}
STAGE=${STAGE:-1}

module load conda/latest
conda activate /work/pi_nwycoff_umass_edu/.conda/envs/hang

SCRIPT_DIR_SELF="$ROOT_DIR/scripts/slurm"
CODE_DIR=$(dirname "$TRAIN_PY")
if [ "$(basename "$CODE_DIR")" = "code" ]; then
    DESIGN_DIR=$(dirname "$CODE_DIR")
else
    DESIGN_DIR="$CODE_DIR"
fi

export PYTHONPATH="$CODE_DIR:$ROOT_DIR:/home/hangyang_umass_edu/MMC/sapiens/pose:${PYTHONPATH:-}"
export BEDLAM2_DATA_ROOT="/work/pi_nwycoff_umass_edu/hang/BEDLAM2subset"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

IS_BASELINE=0
if [ "$(basename "$DESIGN_DIR")" = "baseline" ]; then
    IS_BASELINE=1
fi

echo "============================================================"
echo "[stage${STAGE}] start at $(date) (attempt=${SLURM_RESTART_COUNT:-0}, job=${SLURM_JOB_ID:-?}, node=${SLURMD_NODENAME:-?})"
echo "============================================================"
STAGE=$STAGE python "$TRAIN_PY" || {
    echo "Stage $STAGE exited with code $? at $(date)" > "$DESIGN_DIR/training_failed.txt"
    exit 1
}
echo "[stage${STAGE}] done at $(date)"

# Successful stage → clean checkpoints to save disk (metrics.csv / logs are kept).
STAGE_WORK_DIR="$DESIGN_DIR/output/stage${STAGE}"
if [ -d "$STAGE_WORK_DIR" ]; then
    find "$STAGE_WORK_DIR" -maxdepth 2 -type f \( -name '*.pth' -o -name 'last_checkpoint' \) -delete
    echo "[stage${STAGE}] cleaned checkpoints in $STAGE_WORK_DIR"
fi

# Only stage 1 is responsible for conditionally submitting stage 2.
if [ "$STAGE" != "1" ]; then
    exit 0
fi

STAGE1_METRICS="$DESIGN_DIR/output/stage1/metrics.csv"
BASELINE_STAGE1_METRICS="$ROOT_DIR/runs/baseline/output/stage1/metrics.csv"

if [ "$IS_BASELINE" -ne 1 ]; then
    if [ ! -f "$STAGE1_METRICS" ] || [ ! -f "$BASELINE_STAGE1_METRICS" ]; then
        echo "[gate] stage-1 metrics missing; not submitting stage 2"
        exit 0
    fi
    PASS=$(python - "$STAGE1_METRICS" "$BASELINE_STAGE1_METRICS" <<'PY'
import csv, sys
def last_composite(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    v = rows[-1].get('composite_val', '')
    return float(v) if v not in ('', None) else None
d = last_composite(sys.argv[1])
b = last_composite(sys.argv[2])
if d is None or b is None:
    print('NO')
else:
    print('YES' if d < b else 'NO')
PY
)
    if [ "$PASS" != "YES" ]; then
        echo "[gate] stage 1 did not beat baseline; not submitting stage 2"
        exit 0
    fi
    echo "[gate] stage 1 beat baseline; submitting stage 2"
else
    echo "[baseline] submitting stage 2 unconditionally"
fi

JOB_NAME="$(basename "$(dirname "$DESIGN_DIR")")-$(basename "$DESIGN_DIR")-s2"
# For baseline the names collapse to "runs-baseline-s2"; override to "baseline-s2".
if [ "$IS_BASELINE" -eq 1 ]; then
    JOB_NAME="baseline-s2"
fi
sbatch \
    --job-name="$JOB_NAME" \
    --time=16:00:00 \
    --time-min=02:00:00 \
    --mem=48G \
    -o "$DESIGN_DIR/slurm_s2_%j.out" \
    --open-mode=append \
    --export=ALL,TRAIN_PY="$TRAIN_PY",STAGE=2 \
    "$SCRIPT_DIR_SELF/slurm_train.sh"
