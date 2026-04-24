# Setup Summary

## Target Project
`/home/hangyang_umass_edu/MMC/sapiens/pose` — Sapiens 0.3B RGBD 3D pose estimation on BEDLAM2 data with a transformer decoder head.

## Core Configuration

1. **Training script:** `tools/train.py <config.py> --work-dir <dir>` (run from sapiens/pose dir)
2. **Primary metric:** `composite_val` = 0.67 * mpjpe/body/val + 0.33 * mpjpe/pelvis/val (lower is better)
3. **All tracked metrics:** composite_val, mpjpe_body_val, mpjpe_pelvis_val, mpjpe_rel_val, mpjpe_hand_val, mpjpe_abs_val
4. **Done when:** epoch >= 20
5. **Runtime:** SLURM (partition=gpu-preempt, gres=gpu:2080_ti:1, time=8h stage1 / 16h stage2)
6. **Submit-test:** 1 epoch with max_seqs=5 train/5 val for fast validation (~3 min)

## Training Invariants
- **Batch size:** 4 per GPU, gradient accumulation 8 (effective batch size 32)
- **Seed:** 2026
- **Checkpointing for preemption:** CheckpointHook saves every epoch with `max_keep_ckpts=1` (only the latest `.pth` is kept on disk); config sets `resume=True` so preempted jobs auto-resume from the latest checkpoint in the stage work_dir. On successful stage completion, `slurm_train.sh` deletes the remaining `.pth` / `last_checkpoint` files to save disk (metrics.csv and logs are kept).
- **LR scheduling:** by iteration (not epoch) — so warmup/decay stay consistent when more data is added later
- **Data split:** loaded from JSON at `/work/pi_nwycoff_umass_edu/hang/auto/splits_rome_tracking.json` (100 train / 50 val sequences)
- **Data root:** `/work/pi_nwycoff_umass_edu/hang/BEDLAM2subset`
- **Pretrained checkpoint:** `/home/hangyang_umass_edu/MMC/sapiens/pretrain/checkpoints/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth`
- **Loss restricted to body joints (0-21) + pelvis** — hand/surface joint losses removed from training

## Metrics Output (both written incrementally during training)
- **metrics.csv:** one row per epoch after validation — columns: epoch, composite_val, mpjpe_body_val, mpjpe_pelvis_val, mpjpe_rel_val, mpjpe_hand_val, mpjpe_abs_val
- **iter_metrics.csv:** one row per iteration — columns: iter, epoch, loss_joints_train, loss_depth_train, loss_uv_train

## Baseline Files (experimentable, copied per design)
- `train.py` — wrapper script that calls `tools/train.py` with the co-located config and auto-sets work_dir
- `config.py` — standalone MMEngine config (optimizer, LR schedule, data pipeline, model, hooks)
- `pose3d_transformer_head.py` — transformer decoder head (architecture + loss computation, modified for body-only joint loss)
- `pelvis_utils.py` — pelvis unprojection utilities (imported by head)

## Infra Files (shared, never modified between designs)
- `constants.py` — machine paths, joint indices, research invariants
- `metrics_csv_hook.py` — custom MMEngine hook that writes metrics.csv and iter_metrics.csv

Everything else in `mmpose/` is part of the installed package and stays in the target project (backbone, dataset, transforms, metric, data preprocessor, pose estimator, loss modules).

## SLURM Configuration
- `--partition=gpu-preempt`
- `--gres=gpu:2080_ti:1`
- `--cpus-per-task=1`
- `--mem=24G` (stage 1), `--mem=48G` (stage 2, overridden in slurm_train.sh when submitting stage 2)
- `--time=08:00:00` (stage 1), `--time=16:00:00` (stage 2)
- Conda: `module load conda/latest && conda activate hang`
- `PYTHONPATH=<code_dir>:<repo_root>:/home/hangyang_umass_edu/MMC/sapiens/pose`
- `BEDLAM2_DATA_ROOT=/work/pi_nwycoff_umass_edu/hang/BEDLAM2subset`

## Contract
- **Experimentable files:** config.py, pose3d_transformer_head.py, pelvis_utils.py (architecture, loss, optimizer, LR schedule, augmentation, model hyperparameters)
- **Must never change:** evaluation metric (bedlam_metric.py), dataset (bedlam2_dataset.py), data transforms (bedlam2_transforms.py), backbone (sapiens_rgbd.py), data preprocessor, infra/metrics_csv_hook.py, infra/constants.py, train.py wrapper, tools/train.py

## Preferences
- **Model preferences:** Opus for Architect role
- **Auto GitHub issue filing:** disabled
