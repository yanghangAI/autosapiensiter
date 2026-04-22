**Verdict:** APPROVED

**Summary:** Design B is identical in code to design001 but with `uv_heatmap_sigma=1.0` and `uv_heatmap_loss_weight=1.0`. The delta is entirely in `config.py`. Head/util code is specified by reference to design001, which is acceptable because design001's spec is fully concrete and the referenced code is identical.

**Checks:**
- Design Description present.
- Starting point: `baseline/`.
- Files to change: only `pose3d_transformer_head.py`, `pelvis_utils.py`, `config.py`.
- Algorithmic delta vs design001 clearly and completely specified (sigma, loss weight). All other behaviour inherited verbatim.
- Exact config kwargs and values provided.
- Output contract preserved.
- Invariants preserved; no invariant file modified.
- MMEngine config constraint satisfied (literals only).
- Edge case (sigma=1 fp16 underflow at far corners, renormalization) explicitly addressed.

**Nits (non-blocking):**
- Builder must copy the head code from design001 verbatim — this is stated explicitly in design.md and is unambiguous.

Builder can implement without guessing. Approved.
