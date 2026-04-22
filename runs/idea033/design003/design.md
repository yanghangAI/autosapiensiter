# Design 003 — Variant C: Pelvis-Token FiLM at Output

**Design Description:** Same K extraction, normalization, and zero-init FiLM MLP as Designs 001/002, but FiLM is applied **only to the pelvis token (index 0)** after the decoder layer and **just before** `depth_out` and `uv_out`; body joints and their output Linear are untouched, so the K signal directly modulates only the two K-dependent outputs (`pelvis_depth`, `pelvis_uv`). This matches the causal structure of the problem — body joints are K-invariant root-relative metres, pelvis UV/depth are K-dependent.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — same scaffolding as Designs 001/002; FiLM is applied to the pelvis token only.
2. `config.py` — set `k_film_variant='pelvis'`.
3. `pelvis_utils.py` — unchanged.

No other files are modified.

---

## Algorithm

Identical K extraction and FiLM MLP to Design 001 (same `_KFilmMLP`, same normalization `[fx/W_ref, fy/H_ref, cx/cw, cy/ch, ch/H_ref, cw/W_ref]`, same zero-init).

**FiLM application — Variant C:** after `decoded = self.decoder_layer(queries, spatial)` and the unchanged `joints = self.joints_out(decoded)` line, read the pelvis token and modulate only that one:

```python
pelvis_token = decoded[:, 0, :]  # (B, hidden_dim)

if self.use_k_film and self.k_film_variant == 'pelvis':
    if k_batch is None:
        k_batch = torch.zeros(B, 6, device=feat.device, dtype=feat.dtype)
    else:
        k_batch = k_batch.to(device=feat.device, dtype=feat.dtype)
    gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)   # (B, hidden_dim)
    pelvis_token = pelvis_token * (1.0 + gamma) + beta

pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv    = self.uv_out(pelvis_token)
```

Important ordering: `joints = self.joints_out(decoded)` is computed on the **unmodulated** `decoded`, so the K-FiLM signal does not leak into the body-joint pathway. Only the pelvis head sees K-conditioned features.

### Routing K into `forward()`

Identical to Designs 001/002. `forward()` signature:

```python
def forward(self,
            feats: Tuple[torch.Tensor, ...],
            k_batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
```

`loss()` and `predict()` build `k_batch` via `self._build_k_batch(batch_data_samples, feats[-1].device)` when `self.use_k_film` is True and pass it to `forward`.

### Invariants preserved

- Output dict keys and shapes unchanged: `joints` `(B, 70, 3)`, `pelvis_depth` `(B, 1)`, `pelvis_uv` `(B, 2)`.
- Loss keys unchanged.
- Body-only joint loss (indices 0–21) preserved.
- `_train_mpjpe` / `_train_mpjpe_abs` telemetry preserved — `_train_mpjpe_abs` now benefits from K-aware pelvis depth/UV.
- Zero-init guarantee: at step 0, `(gamma, beta) = (0, 0)` → `pelvis_token * 1 + 0 = pelvis_token` → head bit-for-bit baseline.
- Body joints are not modulated; their output matches baseline at step 0 and, crucially, only the K-dependent pelvis pathway is augmented.
- No changes to optimizer, LR schedule, data pipeline, batch size, AMP, seed.

---

## 1. `pose3d_transformer_head.py` Changes

Same scaffolding as Design 001 (§1a–§1g) — the unified head source code contains three guarded FiLM blocks (`'query'`, `'spatial'`, `'pelvis'`); only the pelvis block executes in this design.

Explicit forward-body structure (builder must preserve this exact ordering):

```python
# spatial/query FiLM blocks (skipped when k_film_variant == 'pelvis')
if self.use_k_film and self.k_film_variant == 'spatial':
    ...   # see Design 002
if self.use_k_film and self.k_film_variant == 'query':
    ...   # see Design 001

decoded = self.decoder_layer(queries, spatial)

joints = self.joints_out(decoded)        # (B, num_joints, 3) — K-UNmodulated

pelvis_token = decoded[:, 0, :]          # (B, hidden_dim)
if self.use_k_film and self.k_film_variant == 'pelvis':
    if k_batch is None:
        k_batch = torch.zeros(B, 6, device=feat.device, dtype=feat.dtype)
    else:
        k_batch = k_batch.to(device=feat.device, dtype=feat.dtype)
    gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)
    pelvis_token = pelvis_token * (1.0 + gamma) + beta

pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv    = self.uv_out(pelvis_token)

return {'joints': joints,
        'pelvis_depth': pelvis_depth,
        'pelvis_uv': pelvis_uv}
```

All other head changes (imports, `_KFilmMLP` class, `__init__` kwargs, `_build_k_batch`, `loss()`/`predict()` routing) are identical to Design 001 §1a–§1g.

---

## 2. `config.py` Changes

In the `head=dict(...)` block, append after `loss_weight_uv=1.0,`:

```python
        # ── Camera-intrinsic FiLM (idea033 / Variant C — pelvis-token FiLM) ──
        use_k_film=True,
        k_film_variant='pelvis',
        k_film_hidden=64,
```

No other changes to `config.py`.

---

## 3. `pelvis_utils.py` Changes

None.

---

## Expected Behavior After Change

- At step 0, identity FiLM → head bit-for-bit baseline.
- Parameter count added: `~33.8K` (identical FiLM MLP to Designs 001/002).
- Only pelvis depth and pelvis UV outputs receive K-conditioned features; body joints remain K-invariant (matches the idea's causal hypothesis).
- Primary target improvement: `mpjpe_pelvis_val` and `mpjpe_abs_val`. `mpjpe_body_val` and `mpjpe_rel_val` should be approximately unchanged or marginally better (body loss still propagates through `joints_out` on unmodulated `decoded`; however, via the FiLM MLP's gradient flowing back through `pelvis_token`, there may be second-order effects on shared backbone/decoder params).
- Baseline recoverable via `use_k_film=False`.
- This is the most targeted of the three variants and the one most aligned with the "K-invariant body vs K-dependent pelvis" causal decomposition described in `idea.md`.

---

## Constraints / Edge Cases

- `pelvis_token = decoded[:, 0, :]` is the existing baseline convention — token index 0 is the pelvis. This must not change.
- The FiLM modulation must come **after** `self.joints_out(decoded)` so body joints are computed from unmodulated `decoded`. Do not refactor the order.
- Gradients from `loss/depth/train` and `loss/uv/train` flow through the FiLM MLP; gradients from `loss/joints/train` do not (except through shared backbone/decoder params).
- Same K-missing fallback, same `k_film_hidden=64`, same normalization constants as Designs 001/002.
- FiLM must apply to the **pre-Linear** pelvis feature (i.e. before `self.depth_out` and `self.uv_out`), not to their outputs; gamma/beta are in `hidden_dim=256` space, not in 1-D or 2-D output space.
