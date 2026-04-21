# Code Review — idea017/design002

**Verdict: APPROVED**

## Automated check
`python scripts/cli.py review-check-implementation runs/idea017/design002` — PASSED.

## Files-changed fidelity
`implementation_summary.md` lists `code/pose3d_transformer_head.py` and `code/config.py`. Both correspond to files required by `design.md`. No extra files were changed. `pelvis_utils.py` and `train.py` are unmodified copies of baseline.

## Architecture — `pose3d_transformer_head.py`

- Identical to design001 in structure: `nn.Embedding(22, 256)`, 2-layer `nn.ModuleList`, `hand_proj = nn.Linear(5632, 144)`.
- `aux_body_loss_weight` is stored as an attribute and the `if self.aux_body_loss_weight > 0.0` branch in `loss()` is now entered with value 0.4.
- Intermediate weight formula: `intermediate_weights = [self.aux_body_loss_weight * (1.0 + 0.5 * k) for k in range(n_inter)]` with `n_inter = len(self._intermediate_outputs) - 1 = 1` (2 layers → 1 intermediate). Yields `[0.4]` — correct.
- `loss/joints_aux_0/train` emitted at weight 0.4 from layer-1 output — matches design.
- `loss/hand_aux/train` emitted at weight 0.1 — matches design.
- All other code paths identical to design001 and correct.

## Config — `config.py`

- `aux_body_loss_weight=0.4` — matches design (only difference from design001).
- `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1` — correct.
- No Python `import` statements — compliant.

## Invariants
All invariant files unmodified. Output shape (B,70,3) preserved. `persistent_workers=False` preserved.

## Test output
- Training completed ("Done training!"), `epoch_1.pth` saved.
- Loss keys at iter 50: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/joints_aux_0/train: 0.085289`, `loss/hand_aux/train: 0.070038` — exactly matches design expectations.
- `grad_norm: 16.037` — healthy.
- Memory 8647 MB — within budget.

## Issues
None.
