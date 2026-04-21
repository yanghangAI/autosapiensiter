## Code Review — idea002/design002

**Verdict: APPROVED**

---

### Pre-check

`python scripts/cli.py review-check-implementation runs/idea002/design002` — PASSED.

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
| Add `pelvis_decoder_type: str = 'shared'` constructor param | PRESENT (line 169) |
| Store both as instance attributes | PRESENT (lines 182–183) |
| `nn.Embedding(1, hidden_dim)` named `pelvis_query` created when `decouple_pelvis=True` | PRESENT (line 200) |
| `_DecoderLayer` named `pelvis_decoder` created when `pelvis_decoder_type='independent'` | PRESENT (lines 201–202) |
| `trunc_normal_(pelvis_query.weight, std=0.02)` in `_init_head_weights` | PRESENT (lines 223–224) |
| Forward: condition is `decouple_pelvis and pelvis_decoder_type == 'independent'` | PRESENT (line 274) |
| Pelvis query uses `pelvis_decoder.norm2`, `.cross_attn`, `.dropout2`, `.norm3`, `.ffn` (cross-attn only, no self-attn) | PRESENT (lines 279–283) |
| `pelvis_token = (pq + pq_ffn)[:, 0, :]` | PRESENT (line 283) |
| Fallback `pelvis_token = decoded[:, 0, :]` for non-independent paths | PRESENT (line 285) |
| `joints_out` still reads all 70 decoded tokens | PRESENT (line 272) |
| Return dict keys `joints`, `pelvis_depth`, `pelvis_uv` unchanged | PRESENT (lines 290–294) |
| `loss()` body-only restriction indices 0–21 | PRESENT (lines 330–331) |
| Absolute imports preserved | PRESENT |
| Docstring updated | PRESENT (lines 22–27) |

Note: design specifies `pelvis_decoder` should be created "after `self.decoder_layer`". In the code, `decoder_layer` is at line 196 and the conditional block is at lines 199–202. This matches the design spec exactly.

The unused `self_attn`, `norm1`, `dropout1` attributes of `pelvis_decoder` are correctly present (they are instantiated as part of `_DecoderLayer`) but not called — as the design permits and documents.

---

### Fidelity Check: `config.py`

| Design requirement | Status |
|---|---|
| `decouple_pelvis=True` added to head dict | PRESENT (line 146) |
| `pelvis_decoder_type='independent'` added to head dict | PRESENT (line 147) |
| All other head params match baseline exactly | CONFIRMED |
| `persistent_workers=False` in both loaders | CONFIRMED |
| `randomness = dict(seed=2026)` | PRESENT |
| No Python `import` statements | CONFIRMED (`__import__` used) |
| `output_dir` set to design002 path | PRESENT |

---

### Invariants

- `pelvis_utils.py`: unchanged (diff clean)
- `train.py`: unchanged (diff clean)
- No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files

---

### Test Output

Test ran to completion (1 epoch, train + val). No runtime errors or crashes.

- Epoch 1 val metrics produced: `composite_val=493.38`, `mpjpe_body_val=429.76`, `mpjpe_pelvis_val=622.54`
- `metrics.csv` correctly written with all required columns
- Memory usage 10624 MB — within 1080 Ti limits (marginally higher than design001 as expected due to extra `_DecoderLayer` parameters)

Test output is consistent with the model being randomly initialised (epoch 1 only, as expected for a reduced test run).

---

### Summary

Implementation fully matches all design requirements. All invariants preserved. Test ran cleanly with no errors.
