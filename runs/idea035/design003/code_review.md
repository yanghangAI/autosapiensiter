# Code Review — idea035/design003 (Spatially-Shuffled Depth Ablation)

**Verdict: APPROVED**

## Checks performed
- `python scripts/cli.py review-check-implementation runs/idea035/design003` — passed.
- `implementation_summary.md` lists exactly the two allowed files; `pelvis_utils.py` is byte-identical to `baseline/pelvis_utils.py`.
- `pose3d_transformer_head.py` (lines 351-393): registered `DepthAblationDataPreprocessor` subclass; `mode=='shuffle'` branch reshapes channel 3 to `(B, H*W)`, draws a fresh `torch.randperm(H*W, device=flat.device)` per sample, applies the permutation to that row, and reshapes back to `(B, H, W)`. Per-sample marginal histogram is preserved exactly; spatial alignment with RGB is destroyed. RGB channels (0..2) are untouched. Defensive guards on shape/tensor presence are in place.
- `config.py` line 140: `data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='shuffle')`. `custom_imports` still imports the head module so the new class is registered. No other config edits.
- Invariants preserved: evaluation metric, dataset, transforms, backbone, data preprocessor file, infra, and `train.py` wrapper untouched. Body-only joint loss preserved. `persistent_workers=False`. Config uses string literals only.
- `test_output/slurm_test_56728567.out`: 1-epoch reduced training reached `[Done training!]` with checkpoint saved; loss values normal (loss/depth/train ~2.73, loss/joints/train ~0.20). `iter_metrics.csv` populated.

Matches design exactly. No issues.
