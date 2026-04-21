# Design Review — idea017/design001

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- Architecture is sound: replacing `nn.Embedding(70, 256)` + single `decoder_layer` with `nn.Embedding(22, 256)` + `nn.ModuleList` of 2 `_DecoderLayer(256, 8, 0.1)` instances is straightforward and consistent with the baseline code structure.
- `hand_proj = nn.Linear(22*256, 48*3)` = `Linear(5632, 144)` is a standard linear layer; no exotic ops.
- Output shape `(B, 70, 3)` is preserved by `torch.cat([body_joints, hand_joints], dim=1)` where `body_joints` is `(B, 22, 3)` and `hand_joints` is `(B, 48, 3)`.
- Storing `self._intermediate_outputs` on `self` during `forward()` and consuming in `loss()` is safe given MMEngine's sequential forward+loss call pattern during training.
- VRAM: 2 layers × 22-query attention is cheaper than 1 layer × 70-query attention (as documented). Feasible on 2080 Ti 8 GB.

### Completeness
- Starting point: `baseline/` — specified.
- Files to change: `pose3d_transformer_head.py` and `config.py` — both fully specified. `pelvis_utils.py` explicitly unchanged.
- All new constructor kwargs specified with types and values: `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0`.
- `__init__` body changes: full code given for all new attributes, the replacement of `self.joint_queries`, the replacement of `self.decoder_layer` with `self.decoder_layers`, and the addition of `self.hand_proj`.
- `_init_head_weights()` addition: exact lines given (`trunc_normal_` on `hand_proj.weight`, `zeros_` on bias).
- `forward()` replacement: full code given, including `intermediate_outputs` collection, `body_flat` reshape, `hand_proj`, `torch.cat`, pelvis token extraction, and `self._intermediate_outputs` assignment.
- `loss()` additions: both the `aux_body_loss_weight > 0.0` branch (inactive for Design 001 since weight=0.0) and the `hand_aux_loss_weight > 0.0` branch are specified with exact code.
- `config.py` head dict: all kwargs listed as literals. No Python import statements. Compliant with MMEngine config constraints.

### Explicitness
- The `_BODY` index list in `loss()` is correctly defined as `list(range(0, 22))` consistent with baseline.
- `_HAND = list(range(22, 70))` for the auxiliary hand loss is explicit and correct (48 joints).
- The note about `joints_out` being shared for all layers' intermediate predictions is explicit.
- `self.num_joints = 70` invariant is documented (constraint 1) so `predict()` needs no change.
- Constraint 7 (safety of `self._intermediate_outputs`) is well-justified.
- The design notes that `decoder_layer` (singular) should be removed and replaced with `decoder_layers` (plural ModuleList), which is the only possible ambiguity in the baseline — this is stated explicitly.

### Invariant Compliance
- No changes to invariant files: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py` — all confirmed unchanged.
- `pelvis_utils.py`: explicitly unchanged.
- Loss restricted to body joints (indices 0-21) for the main body loss — preserved.
- `persistent_workers=False` — constraint documented, config not changed.
- No Python `import` statements in `config.py` — all new head kwargs are literals.

### Issues / Notes
- None. The design is self-consistent, detailed, and leaves nothing for the Builder to guess. The inactive `aux_body_loss_weight` branch (0.0) is included for interface compatibility with Designs 002/003, which is a clean design decision.
