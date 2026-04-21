# Project Summary

## Overview

This automation framework runs an iterative research loop on a **Sapiens 0.3B RGBD 3D pose estimation** model trained on the BEDLAM2 dataset. The goal is to minimize `composite_val` = 0.67 * body_MPJPE + 0.33 * pelvis_MPJPE (lower is better) through systematic architectural and hyperparameter experiments, coordinated by specialized AI agents.

The model takes 4-channel RGBD input, passes it through a Sapiens ViT backbone, and uses a transformer decoder head to regress 70 root-relative 3D joint positions plus absolute pelvis location. Training loss is computed only on the 22 body joints and pelvis (depth + UV), not hands or surface joints.

## Baseline

- **Backbone:** SapiensBackboneRGBD (0.3B params, pretrained), drop_path_rate=0.1
- **Head:** Pose3dTransformerHead (1 decoder layer, 8 attention heads, dropout=0.1)
- **Optimizer:** AdamW (lr=1e-4, backbone lr_mult=0.1, weight_decay=0.03), AMP via `FixedAmpOptimWrapper`, grad_accum=8
- **LR Schedule (per stage):** linear warmup + cosine annealing, iteration-based
- **Loss:** SoftWeightSmoothL1 on body joints (0-21) + pelvis depth + pelvis UV
- **Data:** BEDLAM2 `data600/` subset; splits read from `train100.txt`, `train400.txt`, `val200.txt` at the repo root. `Bedlam2Dataset.load_data_list()` drops frames whose pelvis forward-distance X lies outside `[0.5 m, 12 m]` (≈9% of train, ≈6% of val) — `X ≤ 0` breaks the projection formula and `X > ~12 m` is outside the depth-normalisation clip.
- **Train batch:** 4, **val batch:** 16 (val has no backward pass, so fits larger batches on the same GPU). `num_workers=2`, `pin_memory=True`, `persistent_workers=False`.
- **Validation cadence:** `val_interval=5` — validate at epochs 5/10/15/20 in stage 1 and 5/10 in stage 2, so val cost (`val200` is large) doesn't dominate wall time.

Key baseline files in `baseline/`:
- `config.py` — MMEngine config; stage is selected at runtime via the `STAGE` env var
- `pose3d_transformer_head.py` — transformer decoder head with body-only loss
- `pelvis_utils.py` — pelvis 3D reconstruction utilities
- `train.py` — wrapper that reads `STAGE`, sets `--work-dir=output/stage{N}/`, and invokes `tools/train.py`

## Two-Stage Training Flow

Every design (and the baseline) runs up to two stages, driven by the `STAGE` env var set by the SLURM scripts:

| Stage | Train split  | Epochs | Warmup | val epochs | SLURM time | SLURM `--mem` |
|-------|--------------|--------|--------|------------|------------|---------------|
| 1     | `train100.txt` | 20     | 3      | 5,10,15,20 | 8h         | 24G           |
| 2     | `train400.txt` | 10     | 1      | 5,10       | 16h        | 48G           |

Both stages validate on `val200.txt` and write to `output/stage{N}/metrics.csv`. **Stage 2 starts from scratch** with the same pretrained backbone (no weights are transferred from stage 1).

**Gate:** after stage 1 succeeds, `slurm_train.sh` compares the design's stage-1 `composite_val` against baseline's stage-1 `composite_val`. Stage 2 is submitted only if the design strictly beat baseline. Baseline itself runs stage 2 unconditionally. Designs that are gated out end with a single stage-1 row in `results.csv`.

## Metrics

- **Primary:** `composite_val` = 0.67 * mpjpe_body_val + 0.33 * mpjpe_pelvis_val (mm, lower is better)
- **Tracked:** composite_val, mpjpe_body_val, mpjpe_pelvis_val, mpjpe_rel_val, mpjpe_hand_val, mpjpe_abs_val
- **`results.csv` columns:** `idea_id, design_id, stage, epoch, <metric fields>`. Designs contribute one row per stage.
- **Completion:** stage 1 reaches epoch 20 (and stage 2 reaches epoch 10 when gated). The results aggregator reports status from the highest-stage row it finds per design; stage 2 uses a done-threshold of 10, stage 1 uses 20.
- **Output files:** `metrics.csv` (per-epoch val metrics), `iter_metrics.csv` (per-iter train losses) under each `output/stage{N}/`.

## Runtime

- **Cluster:** UMass Unity HPC
- **GPU:** NVIDIA 2080 Ti (10.6 GiB VRAM) — AMP uses Turing Tensor Cores
- **SLURM:** `partition=gpu-preempt`, `gres=gpu:2080_ti:1`, `cpus-per-task=1` (preempt partition chosen for instant scheduling — 2080_ti has ~190 free GPUs; trade-off: jobs may be preempted, so training auto-resumes from checkpoint)
- **Resumability:** `CheckpointHook` saves every epoch with `max_keep_ckpts=1`; config sets `resume=True` so preempted jobs auto-resume from the latest `.pth` in the stage work_dir. On successful stage completion, `slurm_train.sh` deletes `*.pth` and `last_checkpoint` to save disk (metrics.csv and logs are kept).
- **Account CPU cap:** the `pi_nwycoff_umass_edu` account is limited to ~1000 concurrent CPUs shared with other users; expect `MaxCpuPerAccount` pending holds when the account is saturated.
- **Conda env:** `/work/pi_nwycoff_umass_edu/.conda/envs/hang`
- **Submission:** `scripts/slurm/submit_train.sh <train.py> <job_name>` submits only stage 1. Stage 1 self-submits stage 2 on success if the gate passes.

## Directory Layout

```
autosapiens_iter/
├── baseline/          # Canonical starting code (copied per design)
│   ├── config.py      # MMEngine config; STAGE-aware
│   ├── pose3d_transformer_head.py  # Head architecture + loss
│   ├── pelvis_utils.py             # Pelvis utilities
│   └── train.py                    # Training wrapper (reads STAGE env var)
├── train100.txt       # Stage-1 training sequences (100)
├── train400.txt       # Stage-2 training sequences (400)
├── val200.txt         # Validation sequences (200, used in both stages)
├── infra/             # Shared stable code (never modified)
│   ├── constants.py   # Paths, joint indices, invariants
│   └── metrics_csv_hook.py  # CSV output hook
├── runs/              # Experiment tracker
│   ├── idea_overview.csv
│   ├── baseline/      # Validated baseline test
│   └── idea001/       # (created by research loop)
│       └── design001/
│           ├── code/           # Implementation
│           ├── output/stage1/  # Stage-1 training output (metrics.csv, logs)
│           ├── output/stage2/  # Stage-2 training output (only if gate passed)
│           └── test_output/    # Sanity check output
├── agents/            # Agent prompts
│   ├── Orchestrator/
│   ├── Architect/
│   ├── Designer/
│   ├── Builder/
│   ├── Reviewer/
│   └── Debugger/
├── scripts/           # Automation CLI and submission scripts
│   ├── cli.py
│   └── slurm/
├── .automation.json   # Framework configuration
├── results.csv        # Aggregated experiment results
└── SETUP_SUMMARY.md
```

## How to Start the Research Loop

1. Open a **new** Claude Code session (fresh context).
2. Tell it: `Read agents/Orchestrator/prompt.md and act as the Orchestrator.`
3. Then: `Run the full autonomous research loop.`

The Orchestrator will spawn Architect, Designer, Reviewer, Builder, and Debugger agents as needed. Training jobs run asynchronously on SLURM — the loop does not wait for jobs to finish before proposing new ideas.

## Baseline Test Results

The baseline pipeline was validated via `submit-test` (1 epoch, 5 train/5 val sequences):
- `runs/baseline/test_output/metrics.csv`: epoch=1, composite_val=454.43, mpjpe_body_val=442.55, mpjpe_pelvis_val=478.55
- `runs/baseline/test_output/iter_metrics.csv`: 81 training iterations logged
- No `training_failed.txt` — pipeline works end-to-end
