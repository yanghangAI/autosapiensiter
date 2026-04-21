# Code Review — idea020/design001

**Verdict: APPROVED**

## Checklist

### review-check-implementation
PASS.

### Files Changed
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. PRESENT and correct.
- `code/config.py` — required by design. PRESENT and correct.

No extra files changed. `pelvis_utils.py` and `train.py` are unchanged (verified by diff against baseline).

### Code vs Design Fidelity

**`_temp_scaled_attn` helper function:**
- Present at module level, placed after `_build_2d_sincos_pos_enc`.
- Correct implementation: manual Q/K/V projection from `mha_module.in_proj_weight/in_proj_bias`, reshape to multi-head, scaled dot-product, per-query temperature division before softmax.
- AMP dtype cast: `tau = temperature.clamp(min=0.1).to(attn.dtype).view(1, 1, Nq, 1)` — present and correct.
- Dropout: uses `mha_module.dropout` float attribute — correct.
- Returns `mha_module.out_proj(out)` — correct.

**`_DecoderLayer` changes:**
- Design specified adding `cross_temp` and `self_temp` to `__init__` and storing as attributes. Implementation diverges: temperatures are NOT stored in `_DecoderLayer`; instead they are passed as arguments to `forward()`. This is a valid alternative that avoids duplicate `nn.Parameter` registration. The summary explicitly documents this deviation with justification — acceptable.
- `forward()` accepts `cross_temp` and `self_temp` as optional args defaulting to `None` — backward compatible.
- Cross-attention block: routes through `_temp_scaled_attn` when `cross_temp is not None` — correct.
- Self-attention block: uses standard `self.self_attn(q, q, q)[0]` when `self_temp is None` — correct for design001.

**`Pose3dTransformerHead.__init__`:**
- Four new kwargs: `use_cross_temp`, `use_self_temp`, `temp_log_space`, `temp_reg_weight` with correct defaults (False/False/False/0.0) — correct.
- When `use_cross_temp=True` and `temp_log_space=False`: creates `self.cross_temp = nn.Parameter(torch.ones(num_joints))` — correct.
- `_DecoderLayer` constructed without temperature params — correct per the revised architecture.
- `assert self.decoder_layer.cross_attn._qkv_same_embed_dim` — present.

**`Pose3dTransformerHead.forward()`:**
- Computes `cross_temp = self.cross_temp` when `use_cross_temp=True` and `temp_log_space=False`.
- Passes `cross_temp` and `self_temp=None` to `self.decoder_layer(...)` — correct.

**`loss()`:**
- No temp_reg loss added (correct: `temp_reg_weight=0.0`).
- Body-only joint loss (`_BODY = list(range(0, 22))`) unchanged.
- Pelvis token at index 0 unchanged.

**`config.py`:**
- `use_cross_temp=True`, `use_self_temp=False`, `temp_log_space=False`, `temp_reg_weight=0.0` — all correct literals, no import statements.

### Invariants
- `pelvis_utils.py`: unchanged (diff exits 0).
- `train.py`: unchanged (diff exits 0).
- `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`: not present in code/ dir (not modified).

### Test Output
- First test run (55859710) failed with `ValueError: some parameters appear in more than one parameter group`. This was a pre-fix test that was superseded.
- Second test run (55859755) succeeded: model loaded, 1 epoch of 72 iterations completed, checkpoint saved, training finished cleanly. Loss values are within expected range.
- `iter_metrics.csv`: 72 rows for epoch 1, all three loss columns populated correctly.
