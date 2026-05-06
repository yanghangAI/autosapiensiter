# Code Review — idea035/design001 (Zero Depth Ablation)

**Verdict: APPROVED**

## Checks performed
- `python scripts/cli.py review-check-implementation runs/idea035/design001` — passed.
- `implementation_summary.md` lists exactly the two files allowed by `design.md`: `code/pose3d_transformer_head.py` and `code/config.py`. `pelvis_utils.py` is byte-identical to `baseline/pelvis_utils.py`.
- `pose3d_transformer_head.py`: appended import of `RGBDPoseDataPreprocessor` from `mmpose.models.data_preprocessors.rgbd_data_preprocessor` and a new module-level `DepthAblationDataPreprocessor` class registered via `@MODELS.register_module()` (lines 351-393). Constructor accepts `mode` with assert on `('zero','gauss','shuffle')`. `forward()` calls `super().forward(...)`, defensively skips when inputs are missing/non-tensor/<4D/<4 channels, and for `mode=='zero'` calls `inputs[:, 3:4].zero_()`, leaving RGB channels (0..2) untouched. Other modes implemented but inert for this design.
- `config.py` line 140: `data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='zero')`. `custom_imports` still references `pose3d_transformer_head` so registration triggers automatically. No other config edits.
- Invariants preserved: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `train.py`, `tools/train.py`, `infra/*` are not in the changed set; `persistent_workers=False`; body-only joint loss; MMEngine config is import-free.
- `test_output/slurm_test_56728565.out`: training reached `[Done training!]` after the reduced 1-epoch run, checkpoint saved, no errors. `iter_metrics.csv` populated correctly with the expected columns.

No issues found. Matches design 1:1.
