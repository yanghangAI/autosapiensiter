# Code Review — idea035/design002 (Gaussian-Noise Depth Ablation)

**Verdict: APPROVED**

## Checks performed
- `python scripts/cli.py review-check-implementation runs/idea035/design002` — passed.
- `implementation_summary.md` lists exactly the two allowed files; `pelvis_utils.py` is byte-identical to `baseline/pelvis_utils.py`.
- `pose3d_transformer_head.py` (lines 351-393): registered `DepthAblationDataPreprocessor` subclass; for `mode=='gauss'` does `inputs[:, 3:4] = torch.randn_like(inputs[:, 3:4])`, preserving dtype/device, leaving RGB channels (0..2) untouched. Defensive guards on shape/tensor presence are in place.
- `config.py` line 140: `data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='gauss')`. `custom_imports` still imports the head module so the new class is registered. No other config edits.
- Invariants preserved: no changes to evaluation metric, dataset, transforms, backbone, data preprocessor file, infra, or `train.py` wrapper. Body-only joint loss preserved. `persistent_workers=False`. Config uses string literals only.
- `test_output/slurm_test_56728566.out`: 1-epoch reduced training reached `[Done training!]` with checkpoint saved; loss values normal (loss/depth/train ~2.73, loss/joints/train ~0.20). `iter_metrics.csv` populated.

Matches design exactly. No issues.
