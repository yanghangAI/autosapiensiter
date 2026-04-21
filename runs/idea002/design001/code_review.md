## Code Review — idea002/design001

**Verdict: APPROVED**

---

### Pre-check

`python scripts/cli.py review-check-implementation runs/idea002/design001` — PASSED.

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
| Add `decouple_pelvis: bool = False` constructor param | PRESENT (line 168) |
| Store as `self.decouple_pelvis` | PRESENT (line 181) |
| `nn.Embedding(1, hidden_dim)` named `pelvis_query` created conditionally | PRESENT (lines 194–195) |
| `trunc_normal_(pelvis_query.weight, std=0.02)` in `_init_head_weights` | PRESENT (lines 218–219) |
| Forward: pelvis query uses `decoder_layer.norm2`, `.cross_attn`, `.dropout2`, `.norm3`, `.ffn` (skip self-attn) | PRESENT (lines 274–278) |
| `pelvis_token = (pq + pq_ffn)[:, 0, :]` | PRESENT (line 278) |
| Fallback `pelvis_token = decoded[:, 0, :]` when `decouple_pelvis=False` | PRESENT (line 280) |
| `joints_out` still reads all 70 decoded tokens | PRESENT (line 267) |
| Return dict keys `joints`, `pelvis_depth`, `pelvis_uv` unchanged | PRESENT (lines 285–289) |
| `loss()` body-only restriction indices 0–21 | PRESENT (lines 325–326) |
| Absolute imports preserved | PRESENT |
| Docstring updated | PRESENT (lines 22–27) |

One minor note: design says `pelvis_query` should be created "immediately after `self.joint_queries`" (line 191) and separately states "After `self.decoder_layer`". The code places it between `joint_queries` and `decoder_layer` (lines 193–198). This is consistent with both directions and causes no functional difference.

---

### Fidelity Check: `config.py`

| Design requirement | Status |
|---|---|
| `decouple_pelvis=True` added to head dict | PRESENT (line 146) |
| All other head params match baseline exactly | CONFIRMED |
| `persistent_workers=False` in both loaders | CONFIRMED |
| `randomness = dict(seed=2026)` | PRESENT |
| No Python `import` statements | CONFIRMED (`__import__` used) |
| `output_dir` set to design001 path | PRESENT |

---

### Invariants

- `pelvis_utils.py`: unchanged (diff clean)
- `train.py`: unchanged (diff clean)
- No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files

---

### Test Output

Test ran to completion (1 epoch, train + val). No runtime errors or crashes.

- Epoch 1 val metrics produced: `composite_val=511.04`, `mpjpe_body_val=417.83`, `mpjpe_pelvis_val=700.26`
- `metrics.csv` correctly written with all required columns
- Memory usage 10611 MB — within 1080 Ti limits

Test output is consistent with the model being randomly initialised (epoch 1 only, as expected for a reduced test run).

---

### Summary

Implementation fully matches all design requirements. All invariants preserved. Test ran cleanly with no errors.
