**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16
**Design:** idea004/design002 — Depth sinusoidal encoding per spatial token

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea004/design002` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.md. Present and correct.
- `code/config.py` — required by design.md. Present and correct.

No file changed that was not specified by design.md. `pelvis_utils.py` correctly unchanged.

---

## Code vs Design Fidelity

### `pose3d_transformer_head.py`

1. **New imports** (`numpy`, `torch.nn.functional as F`): present at lines 28-31. Correct.

2. **Module-level `_build_1d_sincos_enc` function** (lines 79-98): matches design spec exactly — takes `(B, N, 1)` depth values and `embed_dim`, asserts even, computes `omega = 1/(10000^(arange/half))`, outer-products depth × omega, concatenates sin and cos. Correct.

3. **`__init__` signature** — `depth_pos_enc_type: str = 'sinusoidal'` at line 185. Correct.

4. **`depth_pos_proj = nn.Linear(depth_enc_in_dim, hidden_dim)`**:
   - `depth_enc_in_dim = hidden_dim + hidden_dim // 2` computed at line 215 (not hardcoded 384). Satisfies design constraint 5.
   - Layer created at line 216.
   - `trunc_normal_(std=0.02)` weight init (line 218), zero bias (line 219). Exact match.

5. **`_extract_depth_map` helper** (lines 253-296): identical to Design A spec. Correct.

6. **`_get_pos_enc` preserved**: present at lines 244-251, used in `forward`. Correct per design (Design B calls `_get_pos_enc` and feeds its output into concat+project pipeline).

7. **`forward` body**:
   - `pos_enc_2d = self._get_pos_enc(H, W, feat.device)` called (line 325). Correct.
   - **`depth_map is not None` branch** (lines 327-335): normalise depth clamp [0,10]/10, `_build_1d_sincos_enc(depth_flat, self.hidden_dim // 2)` — uses `self.hidden_dim // 2` not hardcoded 128 (satisfies constraint 4), expands 2D pos enc to batch, concatenates with depth sine → `(B, H*W, 384)`, projects through `depth_pos_proj` → `(B, H*W, 256)`. Exact match.
   - **Fallback `depth_map is None` branch** (lines 336-343): pads with zero depth sine, concatenates, still passes through `depth_pos_proj`. Design constraint 6 ("fallback is not a skip") satisfied — `depth_pos_proj` always in graph.
   - `spatial = spatial + pos_embed` (line 344). Correct — replaces the old `spatial = spatial + pos_enc` pattern.

8. **`loss` and `predict`**: extract depth before `self.forward`, pass as `depth_map=depth_map`. Exact match. Unchanged downstream code.

9. **Invariants**: body joint loss `_BODY = list(range(0, 22))` (line 403). Pelvis via `decoded[:, 0, :]` (line 356). Both preserved.

### `config.py`

- `depth_pos_enc_type='sinusoidal'` present in head dict (line 146). Correct.
- All other config values identical to baseline. `depth_npy_path` in `meta_keys`. No Python `import` statements.

---

## Invariant File Check

Only `code/pose3d_transformer_head.py` and `code/config.py` changed. No invariant files modified. Correct.

---

## Test Output

- SLURM job `55635413` completed successfully with "[test] Finished."
- Training ran 1 epoch, seed 2026, GPU GTX 1080 Ti. No errors.
- All expected metrics produced. `metrics.csv` correct.
- Epoch-1 train loss (1.62) and val composite (456.75 mm) finite and plausible.

---

## Summary

All design requirements fully and correctly implemented. No deviations found. Test run clean.
