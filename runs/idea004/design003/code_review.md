**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16
**Design:** idea004/design003 — Depth+2D MLP positional encoding (NeRF-style)

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea004/design003` — PASSED.

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

2. **`__init__` signature** — `depth_pos_enc_type: str = 'mlp'` at line 163. Correct.

3. **`pos_mlp` architecture** (lines 193-197):
   - `pos_mlp_hidden = 64` hardcoded local variable (line 192). Correct per constraint 4 (not exposed as config kwarg).
   - `nn.Sequential(nn.Linear(3, pos_mlp_hidden), nn.GELU(), nn.Linear(pos_mlp_hidden, hidden_dim))`. Exact match to `Linear(3,64) → GELU → Linear(64,hidden_dim)` specification.
   - Init loop (lines 199-202): `trunc_normal_(std=0.02)` for weights, `zeros_` for biases on all Linear layers. Exact match.

4. **`_pos_enc_hw` retained**: present at line 211 (commented as "retained for completeness; not called in forward for Design C"). `_get_pos_enc` method present but not called in `forward`. Satisfies design constraints 3 and 9.

5. **`_extract_depth_map` helper** (lines 237-280): identical to Design A/B spec. Correct.

6. **`_build_3d_pos_grid` helper** (lines 282-324):
   - `torch.linspace(-1.0, 1.0, h)` and `torch.linspace(-1.0, 1.0, w)` for x/y (constraint 5 — `[-1, 1]` range). Correct.
   - `indexing='ij'` for meshgrid. Correct.
   - `depth_map is not None` path: `depth_map.flatten(2).squeeze(1)` → `(B, h*w)`, clamp [0,10]/10. Correct.
   - `depth_map is None` path: `B=1`, `torch.full((1, h*w), 0.5, device=device)`. Fallback depth = 0.5 per constraint 7. Returns `(1, h*w, 3)` per constraint 8. Correct.
   - Stacks `[xx_batch, yy_batch, depth_flat]` → `(B, h*w, 3)`. Correct.

7. **`forward` body**:
   - `_get_pos_enc` NOT called anywhere in `forward` (constraint 9 — no `pos_enc` addition). Verified by inspection of lines 345-378.
   - `pos_grid = self._build_3d_pos_grid(H, W, depth_map, feat.device)` (line 353).
   - `if depth_map is None: pos_grid = pos_grid.expand(B, -1, -1)` (lines 355-356) — expands `(1, H*W, 3)` to `(B, H*W, 3)`. Satisfies constraint 8.
   - `pos_embed = self.pos_mlp(pos_grid)` (line 357) → `(B, H*W, hidden_dim)`.
   - `spatial = spatial + pos_embed` (line 358). Correct.
   - No `spatial = spatial + pos_enc` line present. Constraint 9 satisfied.

8. **`loss` and `predict`**: extract depth before `self.forward`, pass as `depth_map=depth_map`. Exact match.

9. **Invariants**: body joint loss `_BODY = list(range(0, 22))` (line 417). Pelvis via `decoded[:, 0, :]` (line 370). Both preserved.

### `config.py`

- `depth_pos_enc_type='mlp'` present in head dict (line 146). Correct.
- All other config values identical to baseline. `depth_npy_path` in `meta_keys`. No Python `import` statements.

---

## Invariant File Check

Only `code/pose3d_transformer_head.py` and `code/config.py` changed. No invariant files modified. Correct.

---

## Test Output

- SLURM job `55635414` completed successfully with "[test] Finished."
- Training ran 1 epoch, seed 2026, GPU GTX 1080 Ti. No errors.
- Note: design003 ran significantly slower (iter time ~15s vs ~1.8s for designs 001/002). This is consistent with design expectation — `_build_3d_pos_grid` is called per-forward-pass and constructs tensors on GPU, but the primary overhead appears to be the depth loading per sample. This is a known risk flagged in design.md ("Training speed" section). Not a correctness issue; full training will expose whether this is a bottleneck.
- All expected metrics produced. `metrics.csv` correct.
- Epoch-1 val composite (462.19 mm) finite and plausible.

---

## Summary

All design requirements fully and correctly implemented. No deviations found. Test run clean. Runtime slowdown noted but expected and documented in design.
