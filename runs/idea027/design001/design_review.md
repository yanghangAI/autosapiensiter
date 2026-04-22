## Design Review — idea027/design001

**Verdict: APPROVED**

---

### Feasibility

The design is architecturally sound. The baseline `forward()` already has `B, C, H, W = feat.shape` and the `spatial = spatial + pos_enc` line is a clearly identifiable insertion point. The depthwise-separable conv architecture is standard PyTorch and operates on AMP-safe modules only. No new dependencies are required.

---

### Completeness and Explicitness

All required fields are present and unambiguous:

- **Design Description:** present and accurate.
- **Starting point:** `baseline/` — explicit.
- **Files changed:** `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly excluded.
- **Algorithmic change:** fully specified. The `_SpatialContextNet` class is provided verbatim with exact layer ordering, padding formula (`kernel_size // 2`), initialization calls, and the `forward()` reshape logic.
- **Insertion point in `forward()`:** "After `spatial = spatial + pos_enc` and before `queries = self.joint_queries...`" — unambiguous given the baseline source.
- **Insertion point in `__init__()`:** "after `self.loss_weight_uv = loss_weight_uv`" for the flag store, and "after `self.decoder_layer = _DecoderLayer(...)`" for the conditional instantiation — both are unique lines in the baseline.
- **Config values:** all six kwargs listed as literals (`True`, `3`, `1`, `'none'`, `'gelu'`). `spatial_ctx_groups` intentionally omitted from config since `norm='none'` — the design explicitly notes this in the parameter table. The Builder defaults in the head signature cover the unused arg (`spatial_ctx_groups: int = 32`).
- **Init strategy:** zero-init on pointwise weight and bias via `nn.init.zeros_`. Depthwise via `kaiming_normal_`. Explicitly stated.
- **Exact module structure for num_layers=1:** dw → Identity → GELU → pw. Unambiguous from the class loop.

---

### Invariants Audit

1. Zero-init guarantee clearly stated and mechanically correct: for `num_layers=1`, `is_last=True` at `i=0`, so `zero_init_last=True` triggers zeros on the only pointwise. Delta=0 at init. Correct.
2. Shape invariant: `(B, H*W, hidden_dim)` preserved by the reshape → net → reshape → transpose chain.
3. `H, W` sourced from `feat.shape` — correct.
4. Config constraint: all literals, no imports. Satisfied.
5. Loss/output interfaces: unchanged (only `forward()` modified, before decoder).
6. `persistent_workers=False`: not touched.
7. AMP safety: `Conv2d`, `GELU`, `Identity` — all safe.

---

### No Invariant Violations

The design does not modify: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.

---

### Minor Notes (non-blocking)

- The design's parameter table lists `spatial_ctx_groups` as "not passed (unused when norm='none')" — the Builder should still include the kwarg in `__init__` with default `32` as specified, even though it is not passed from config. The design makes this clear.
- The `zero_init_last=True` is hardcoded at the call site in `__init__`, not passed from config. This is correct and consistent with the design.
