#!/bin/bash
#SBATCH --job-name=test_train
#SBATCH --partition=gpu-preempt
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00

set -u

# TARGET_DIR and ROOT_DIR are passed via --export from the sbatch call
TARGET_DIR=${TARGET_DIR:-${1:-$PWD}}
ROOT_DIR=${ROOT_DIR:-/work/pi_nwycoff_umass_edu/hang/autosapiens_iter}

cd "$TARGET_DIR/code" || exit 1

module load conda/latest
conda activate /work/pi_nwycoff_umass_edu/.conda/envs/hang

echo "[test] Running test train in $TARGET_DIR"

# PYTHONPATH: code dir (for local modules), repo root (for infra), sapiens pose
export PYTHONPATH="$TARGET_DIR/code:$ROOT_DIR:/home/hangyang_umass_edu/MMC/sapiens/pose:${PYTHONPATH:-}"
export BEDLAM2_DATA_ROOT="/work/pi_nwycoff_umass_edu/hang/BEDLAM2subset"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Override for test: 1 epoch, max_seqs=5 for train and val
TEST_OUTPUT="$TARGET_DIR/test_output"
mkdir -p "$TEST_OUTPUT"

python /home/hangyang_umass_edu/MMC/sapiens/pose/tools/train.py \
    config.py \
    --work-dir "$TEST_OUTPUT" \
    --cfg-options \
    train_cfg.max_epochs=1 \
    train_dataloader.dataset.max_seqs=5 \
    val_dataloader.dataset.max_seqs=5 \
    || {
    echo "Test training exited with code $? at $(date)" > "$TARGET_DIR/training_failed.txt"
    exit 1
}

rm -f "$TEST_OUTPUT"/*.pth
echo "[test] Finished."
