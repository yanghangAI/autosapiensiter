## Design Review — idea003 / design001

**Verdict: APPROVED**

---

### Checklist

**Design Description present:** Yes — "Single-linear global conditioning on joint queries (minimal additive offset from mean-pooled spatial tokens)."

**Starting point specified:** Yes — `baseline/`.

**Files to modify:** `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly not modified. No invariant files touched.

**Algorithmic change — exact and unambiguous:** Yes.
- New `__init__` parameter `query_cond_type: str = 'linear'` added after `init_cfg`. ✓
- `self.query_cond_net = nn.Linear(hidden_dim, num_joints * hidden_dim)`. ✓
- Init: `trunc_normal_(weight, std=0.02)`, `zeros_(bias)`. ✓
- `query_cond_type` stored as `self.query_cond_type`. ✓
- `else: raise ValueError(...)` guard present. ✓
- `_init_head_weights` must NOT reinitialise `query_cond_net`. Explicitly stated. ✓

**Forward change — exact and unambiguous:** Yes.
- `global_feat = spatial.mean(dim=1)` computed after `spatial = spatial + pos_enc`. Explicitly stated as a constraint. ✓
- `offsets = self.query_cond_net(global_feat)` then `.reshape(B, self.num_joints, self.hidden_dim)`. ✓
- `queries = static_q + offsets` before calling `self.decoder_layer`. ✓
- `expand` vs `repeat` note included (safe with an additive new tensor). ✓

**Config change:** `query_cond_type='linear'` added to the `head` dict. All other values identical to baseline. ✓

**Parameter count stated:** ~4.61 M additional params, ~5% overhead. No extra attention operations. ✓

**Invariants preserved:**
- Loss restricted to body joints 0-21. Explicitly confirmed. ✓
- Pelvis pathway `decoded[:, 0, :]` unchanged. Explicitly confirmed. ✓
- `persistent_workers=False` unchanged. ✓
- Seed 2026 unchanged. ✓
- No import statements in config (plain string literal). ✓
- Absolute imports in head file unchanged (not a new import needed). ✓

**Edge cases / constraints:** All documented — zero-bias init for near-zero start, positional-encoding placement, expand-vs-repeat safety. No ambiguity for the Builder.

**Implementation readiness:** A Builder can implement this without guessing. The before/after code blocks are exact and complete. The placement within `__init__` ("after the line that creates `self.decoder_layer`") is unambiguous given the baseline source.

---

**No issues found.**
