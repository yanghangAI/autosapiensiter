**Design Review Verdict: APPROVED**

**Design ID:** idea016/design003
**Reviewer:** Reviewer Agent
**Date:** 2026-04-21

---

## Summary

Design 003 implements hierarchical spatial-block FiLM conditioning (4×4=16 blocks, shared MLP) as Design C from idea.md. The specification is detailed, mathematically verified step-by-step, and implementation-ready despite being the most complex of the three designs.

---

## Checklist

### Feasibility
- PASS. The spatial block decomposition is exact: H'=40 ÷ 4 = 10 (row-block size), W'=24 ÷ 4 = 6 (col-block size), 16 blocks × 60 tokens = 960 tokens. Integer division is clean with no remainder.
- PASS. Token layout is correctly identified: `feat.flatten(2).transpose(1,2)` produces row-major token ordering (i = row × W' + col). The `spatial.view(B, H, W, D).view(B, 4, H//4, 4, W//4, D)` reshape correctly partitions the contiguous row-major grid into 4×4 spatial blocks.
- PASS. The step-by-step reshape trace is provided and verified:
  - `(B, 960, 256)` → `(B, 40, 24, 256)` → `(B, 4, 10, 4, 6, 256)` → mean(dim=2) → `(B, 4, 4, 6, 256)` → mean(dim=3) → `(B, 4, 4, 256)` — dimensions correct at each step.
- PASS. The scatter-back logic is fully specified: `gamma_b.unsqueeze(2).expand(B, 4, H//4, 4, W//4, D)` → `.reshape(B, 960, D)`. Crucially, `expand` does not allocate memory (lazy view), and `reshape` after `expand` will allocate — this memory implication is explicitly called out.
- PASS. PyTorch `nn.Linear` broadcasting over leading dims enables the shared MLP to be applied to `(B, 4, 4, hidden_dim)` without loops — correctly stated.
- PASS. `B`, `H`, `W` variables are already in scope from `B, C, H, W = feat.shape` in the existing `forward()` — no new extraction needed.

### Completeness
- PASS. Starting point is `baseline/` — explicit.
- PASS. Files changed: `pose3d_transformer_head.py` and `config.py`. Both fully described.
- PASS. Invariant files explicitly listed as unchanged.
- PASS. Three new constructor kwargs: `film_pool_type: str = 'none'`, `film_hidden_dim: int = 128`, `film_num_blocks: int = 16`. All with defaults.
- PASS. `self.film_num_blocks` stored explicitly (unlike design001/002 which do not store `film_hidden_dim`).
- PASS. Lazy block-shape attributes `_film_block_h`, `_film_block_w` specified (though the design notes these are not actually used since block sizes are hardcoded from known H'/W' — this is a minor redundancy but not an error).
- PASS. `film_net` architecture: `Linear(256, 128) → GELU → Linear(128, 512)` — same as design001, ~98K parameters, shared across 16 blocks.
- PASS. Full `forward()` block provided as exact code with comments.
- PASS. Config additions: `film_pool_type='spatial_block'`, `film_hidden_dim=128`, `film_num_blocks=16` as str/int literals.
- PASS. Full updated `head` dict shown.
- PASS. Alternative one-liner `spatial_blocks.mean(dim=(2, 4))` provided as equivalent option — Builder may use either.
- PASS. AMP compatibility for all ops (view, mean, unsqueeze, expand, reshape, chunk, multiply, add) confirmed.

### Explicitness
- PASS. No ambiguity in reshape logic — step-by-step verification eliminates guessing.
- PASS. Note on `film_num_blocks=16` being accepted as kwarg for interface consistency but implementation hardcoding 4×4 layout: explicitly stated with caveats. Builder knows not to dynamically compute block dims from this kwarg.
- PASS. Behaviour at step 0 explicitly verified.
- PASS. `mean(dim=2).mean(dim=3)` sequential reduction vs. `mean(dim=(2,4))` both specified — no ambiguity.

### Invariants Not Violated
- PASS. No modification to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or `train.py` wrapper.
- PASS. Loss restricted to body joints 0-21 unchanged.
- PASS. `persistent_workers=False`, `resume=True`, `max_keep_ckpts=1`, seed 2026, AMP unchanged.
- PASS. Config uses str/int literals only — no Python imports.

---

## Issues Found

**Minor (non-blocking):** The design declares `self._film_block_h: int | None = None` and `self._film_block_w: int | None = None` but never uses them in `forward()` (the block sizes are hardcoded as `H // 4` and `W // 4`). This creates dead constructor code. However, since the design explicitly notes "These are set lazily in forward() based on actual H', W'" but then never actually sets them (the forward code uses `H // 4` directly), the Builder may either omit these lines or include them as documentation. This is not a blocking issue — the forward logic is fully specified and correct.

---

## Notes

Design 003 is the most complex of the three but is specified with exceptional precision. The reshape verification trace removes the main risk of getting tensor dimension ordering wrong. The shared MLP across 16 block contexts via PyTorch `nn.Linear` broadcasting is a standard pattern (equivalent to a 1×1 convolution) and will work correctly. The constraint that `H//4=10` and `W//4=6` produce exact integer divisions is verified by the design (40÷4=10, 24÷4=6).
