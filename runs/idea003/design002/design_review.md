## Design Review — idea003 / design002

**Verdict: APPROVED**

---

### Checklist

**Design Description present:** Yes — "Two-layer bottleneck MLP global conditioning on joint queries (hidden_dim → 128 → num_joints*hidden_dim, no LayerNorm on offsets)."

**Starting point specified:** Yes — `baseline/`.

**Files to modify:** `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly not modified. No invariant files touched.

**Algorithmic change — exact and unambiguous:** Yes.
- New `__init__` parameter `query_cond_type: str = 'mlp'` added after `init_cfg`. ✓
- `bottleneck_dim = hidden_dim // 2` (= 128 at default). Proportional if `hidden_dim` changes. ✓
- `nn.Sequential(nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, num_joints * hidden_dim))`. ✓
- Exact layer shapes stated: `(128, 256)` and `(17920, 128)`. ✓
- Init loop: `isinstance(layer, nn.Linear)` skip pattern for GELU. ✓
- All biases `zeros_`, all weights `trunc_normal_(std=0.02)`. ✓
- `else: raise ValueError(...)` guard present. ✓
- `_init_head_weights` not modified — explicitly stated. ✓

**Forward change — exact and unambiguous:** Yes.
- Identical pattern to design001: mean-pool after pos_enc, reshape, add to static queries. ✓
- `global_feat` computed after `spatial = spatial + pos_enc`. Explicitly stated as a constraint. ✓
- No LayerNorm on offsets — explicitly noted as intentional distinction from design003. ✓

**Config change:** `query_cond_type='mlp'` added to the `head` dict. All other values identical to baseline. ✓

**Parameter count stated:** ~2.34 M additional (vs. ~4.61 M for design001). Exact breakdown given. ✓

**Invariants preserved:**
- Loss restricted to body joints 0-21. Explicitly confirmed. ✓
- Pelvis pathway `decoded[:, 0, :]` unchanged. Explicitly confirmed. ✓
- `persistent_workers=False` unchanged. ✓
- Seed 2026 unchanged. ✓
- No import in config. GELU is `nn.GELU()` (available via `torch.nn`). ✓
- No dropout inside `query_cond_net`. Explicitly stated. ✓

**Edge cases / constraints:** All documented — bottleneck dimension derivation, GELU vs ReLU choice, Sequential iteration pattern, zero-bias init for near-zero start. No ambiguity for the Builder.

**Implementation readiness:** A Builder can implement this without guessing. Before/after code blocks are exact and complete. Placement in `__init__` identical to design001 ("after the line that creates `self.decoder_layer`").

---

**No issues found.**
