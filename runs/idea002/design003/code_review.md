## Code Review — idea002/design003

**Verdict: APPROVED**

---

### Pre-check

`python scripts/cli.py review-check-implementation runs/idea002/design003` — PASSED.

---

### Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT.

No extra files changed. `pelvis_utils.py` and `train.py` are byte-identical to baseline — confirmed via diff.

---

### Fidelity Check: `pose3d_transformer_head.py`

| Design requirement | Status |
|---|---|
| Add `decouple_pelvis: bool = False` constructor param | PRESENT (line 170) |
| Add `pelvis_decoder_type: str = 'shared'` constructor param | PRESENT (line 171) |
| Store both as instance attributes | PRESENT (lines 184–185) |
| `nn.Embedding(1, hidden_dim)` named `pelvis_query` when `decouple_pelvis=True` | PRESENT (line 202) |
| `_DecoderLayer` named `pelvis_decoder` when `pelvis_decoder_type in ('independent', 'depth_fused')` | PRESENT (lines 203–204) |
| `nn.Linear(hidden_dim, hidden_dim)` named `depth_proj` when `pelvis_decoder_type == 'depth_fused'` | PRESENT (lines 205–206) |
| `trunc_normal_(pelvis_query.weight, std=0.02)` in `_init_head_weights` | PRESENT (lines 226–227) |
| `trunc_normal_(depth_proj.weight, std=0.02)` + `zeros_(depth_proj.bias)` in `_init_head_weights` | PRESENT (lines 229–232) |
| Forward: `global_depth = spatial.mean(dim=1, keepdim=True)` then `depth_proj(global_depth)` | PRESENT (lines 286–287) |
| `spatial_with_depth = torch.cat([global_depth, spatial], dim=1)` | PRESENT (line 290) |
| Pelvis cross-attn uses `spatial_with_depth`, not `spatial` | PRESENT (line 295) |
| `spatial` is NOT modified for joint pathway | CONFIRMED — `decoder_layer(queries, spatial)` at line 277 uses original `spatial` |
| Pelvis decoder uses `pelvis_decoder.norm2`, `.cross_attn`, `.dropout2`, `.norm3`, `.ffn` (skip self-attn) | PRESENT (lines 294–298) |
| `pelvis_token = (pq + pq_ffn)[:, 0, :]` | PRESENT (line 298) |
| Fallback `pelvis_token = decoded[:, 0, :]` for non-depth_fused paths | PRESENT (line 300) |
| Condition in forward is `decouple_pelvis and pelvis_decoder_type == 'depth_fused'` | PRESENT (line 283) |
| `joints_out` reads all 70 decoded tokens | PRESENT (line 280) |
| Return dict keys `joints`, `pelvis_depth`, `pelvis_uv` unchanged | PRESENT (lines 305–309) |
| `loss()` body-only restriction indices 0–21 | PRESENT (lines 345–346) |
| Absolute imports preserved | PRESENT |
| Docstring updated | PRESENT (lines 22–29) |

---

### Fidelity Check: `config.py`

| Design requirement | Status |
|---|---|
| `decouple_pelvis=True` added to head dict | PRESENT (line 146) |
| `pelvis_decoder_type='depth_fused'` added to head dict | PRESENT (line 147) |
| All other head params match baseline exactly | CONFIRMED |
| `persistent_workers=False` in both loaders | CONFIRMED |
| `randomness = dict(seed=2026)` | PRESENT |
| No Python `import` statements | CONFIRMED (`__import__` used) |
| `output_dir` set to design003 path | PRESENT |

---

### Invariants

- `pelvis_utils.py`: unchanged (diff clean)
- `train.py`: unchanged (diff clean)
- No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files

---

### Test Output

Test ran to completion (1 epoch, train + val). No runtime errors or crashes.

- Epoch 1 val metrics produced: `composite_val=523.33`, `mpjpe_body_val=433.03`, `mpjpe_pelvis_val=706.66`
- `metrics.csv` correctly written with all required columns
- Memory usage 10625 MB — within 1080 Ti limits (slightly higher than design002 as expected for `depth_proj` + `spatial_with_depth` tensor)

Test output is consistent with the model being randomly initialised (epoch 1 only, as expected for a reduced test run).

---

### Summary

Implementation fully matches all design requirements. All invariants preserved. Test ran cleanly with no errors.
