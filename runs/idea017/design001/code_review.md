# Code Review — idea017/design001

**Verdict: APPROVED**

## Automated check
`python scripts/cli.py review-check-implementation runs/idea017/design001` — PASSED.

## Files-changed fidelity
`implementation_summary.md` lists `code/pose3d_transformer_head.py` and `code/config.py`. Both correspond to files required by `design.md`. No extra files were changed. `pelvis_utils.py` and `train.py` are unmodified copies of baseline (verified by diff).

## Architecture — `pose3d_transformer_head.py`

- `joint_queries = nn.Embedding(22, 256)` — matches design (22 body-only queries).
- `decoder_layers = nn.ModuleList([_DecoderLayer(256, 8, 0.1), _DecoderLayer(256, 8, 0.1)])` — 2 layers, matches design.
- `hand_proj = nn.Linear(22*256, 48*3)` — matches design exactly.
- `_init_head_weights()`: `trunc_normal_(hand_proj.weight, std=0.02)` and `zeros_(hand_proj.bias)` — matches design.
- `forward()`: iterates `decoder_layers`, collects `intermediate_outputs`, computes `body_joints = joints_out(queries)` (B,22,3), `hand_joints = hand_proj(body_flat)` (B,48,3), concatenates to (B,70,3), extracts `pelvis_token = queries[:, 0, :]` — all match design.
- `self._intermediate_outputs` stored on self for loss() — matches design.
- `loss()`: primary body loss on `pred['joints'][:, _BODY]` vs `gt_joints[:, _BODY]`; `aux_body_loss_weight=0.0` branch correctly skipped; `hand_aux_loss_weight=0.1` branch active, emits `loss/hand_aux/train` — matches design.
- `predict()` unchanged — correct.

## Config — `config.py`

- `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0` — all match design.
- All other values (optimizer, LR schedule, batch size, seed, hooks) unchanged from baseline — correct.
- No Python `import` statements in config — compliant.

## Invariants
- `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `infra/metrics_csv_hook.py`, `train.py` not modified — verified.
- `num_joints=70` preserved in config and code — correct.
- Output shape `(B, 70, 3)` produced by `cat([body_joints, hand_joints], dim=1)` — correct.
- `persistent_workers=False` — preserved.

## Test output
- Training ran to completion ("Done training!"), checkpoint `epoch_1.pth` saved.
- Loss keys logged at iter 50: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train` — exactly as design specifies. No `loss/joints_aux_*/train` keys (correct: `aux_body_loss_weight=0.0`).
- `grad_norm: 16.481` — healthy, no anomalies.
- `iter_metrics.csv` written with 72 rows (1 epoch) — confirmed.
- Memory at iter 50: 8647 MB — within 2080 Ti budget.

## Issues
None.
