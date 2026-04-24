#!/bin/bash

# Submit a stage-1 training job. Stage 2 is submitted from inside
# slurm_train.sh when stage 1 finishes successfully AND the gate passes
# (or always, for the baseline).

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <path_to_train_script> [job_name]"
    echo "Example: $0 runs/baseline/code/train.py baseline"
    exit 1
fi

TRAIN_PY_PATH=$(realpath "$1")
CODE_FOLDER=$(dirname "$TRAIN_PY_PATH")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ "$(basename "$CODE_FOLDER")" = "code" ]; then
    DESIGN_FOLDER=$(dirname "$CODE_FOLDER")
else
    DESIGN_FOLDER="$CODE_FOLDER"
fi

JOB_NAME=${2:-"train_job"}

sbatch \
    --job-name="${JOB_NAME}-s1" \
    --time=08:00:00 \
    --time-min=01:00:00 \
    -o "$DESIGN_FOLDER/slurm_s1_%j.out" \
    --open-mode=append \
    --export=ALL,ROOT_DIR="$SCRIPT_DIR/../..",TRAIN_PY="$TRAIN_PY_PATH",STAGE=1 \
    "$SCRIPT_DIR/slurm_train.sh"
