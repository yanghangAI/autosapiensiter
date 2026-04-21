## Design Review — idea003 / design003

**Verdict: APPROVED**

---

### Checklist

**Design Description present:** Yes — "Two-layer bottleneck MLP global conditioning on joint queries with LayerNorm applied to per-joint offsets before addition."

**Starting point specified:** Yes — `baseline/`.

**Files to modify:** `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly not modified. No invariant files touched.

**Algorithmic change — exact and unambiguous:** Yes.
- New `__init__` parameter `query_cond_type: str = 'mlp_norm'` added after `init_cfg`. ✓
- `bottleneck_dim = hidden_dim // 2 = 128`. ✓
- `nn.Sequential(nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, num_joints * hidden_dim))` — identical to design002. ✓
- `self.query_cond_norm = nn.LayerNorm(hidden_dim)` added as a separate attribute. ✓
- MLP init: `trunc_normal_(std=0.02)` for weights, `zeros_` for biases. ✓
- LayerNorm init: **default PyTorch init** (weight=1.0, bias=0.0). Explicitly not overridden. ✓
- `else: raise ValueError(...)` guard present. ✓
- `_init_head_weights` not modified — explicitly stated. ✓

**Forward change — exact and unambiguous:** Yes.
- `global_feat = spatial.mean(dim=1)` after pos_enc. ✓
- Offsets reshaped to `(B, num_joints, hidden_dim)` **before** applying `query_cond_norm`. ✓
- `self.query_cond_norm(offsets)` applied to the reshaped 3D tensor — LayerNorm normalises over the last dim (hidden_dim=256), independently per (batch, joint) pair. Explicitly stated. ✓
- `queries = static_q + offsets` after normalisation. ✓

**LayerNorm placement constraint:** Explicitly stated and correct — norm must be applied to the **reshaped** `(B, num_joints, hidden_dim)` tensor, not the flat `(B, num_joints * hidden_dim)` output. The Builder cannot get this wrong given the explicit constraint. ✓

**Config change:** `query_cond_type='mlp_norm'` added to the `head` dict. All other values identical to baseline. ✓

**Parameter count stated:** ~2.34 M additional (negligibly more than design002 due to 512 LayerNorm params). ✓

**Init behaviour note:** The design correctly notes that at init, LayerNorm on trunc_normal_(std=0.02) weights may not produce exactly zero offsets (LN normalises to unit variance), but the effect is small and does not impede training start. This is an acceptable and honest clarification. ✓

**Invariants preserved:**
- Loss restricted to body joints 0-21. ✓
- Pelvis pathway `decoded[:, 0, :]` unchanged. ✓
- `persistent_workers=False` unchanged. ✓
- Seed 2026 unchanged. ✓
- No import in config (plain string literal). ✓
- No dropout inside `query_cond_net`. ✓

**Implementation readiness:** A Builder can implement this without guessing. Before/after code blocks are exact and complete. All constraints are explicit and unambiguous.

---

**No issues found.**
