# Design 002 — Variant B: Spatial-Token FiLM from Normalized K

**Design Description:** Same K extraction, normalization, and zero-init FiLM MLP as Design 001, but FiLM is applied to the **projected spatial tokens** (after `input_proj` and addition of the 2D positional encoding) as `s <- (1+gamma)*s + beta`, modulating the key/value source of cross-attention rather than the queries; broadcasts one `(gamma, beta)` per sample across all `H'*W'=40*24=960` spatial tokens.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — same scaffolding as Design 001 (FiLM MLP class, `_build_k_batch` helper, `forward` signature change, `loss/predict` updates). The only functional difference is **where** FiLM is applied.
2. `config.py` — set `k_film_variant='spatial'` instead of `'query'`.
3. `pelvis_utils.py` — unchanged.

No other files are modified.

---

## Algorithm

Identical K extraction and FiLM MLP to Design 001 (same `_KFilmMLP` class, same normalization, same zero-init).

**FiLM application — Variant B:** inside `forward()`, after:

```python
spatial = feat.flatten(2).transpose(1, 2)
spatial = self.input_proj(spatial)
pos_enc = self._get_pos_enc(H, W, feat.device)
spatial = spatial + pos_enc
```

and **before** the decoder layer call `decoded = self.decoder_layer(queries, spatial)`, insert:

```python
if self.use_k_film and self.k_film_variant == 'spatial':
    if k_batch is None:
        k_batch = torch.zeros(B, 6, device=feat.device, dtype=feat.dtype)
    else:
        k_batch = k_batch.to(device=feat.device, dtype=feat.dtype)
    gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)        # (B, hidden_dim)
    spatial = spatial * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
```

Broadcasting `gamma.unsqueeze(1)` with shape `(B, 1, hidden_dim)` across `spatial` shape `(B, H*W=960, hidden_dim)` applies the same per-sample affine to every spatial token — a single K signal modulates the cross-attention's key/value source. The decoder's cross-attention then attends from K-invariant queries to K-conditioned spatial tokens.

The query-branch path is **not** executed in this design: the `if self.k_film_variant == 'query'` block from Design 001's code is present in the file (single unified head) but guarded by the variant string and therefore skipped.

Everything else — joint loss (body-only 0–21), UV loss, depth loss, pelvis-token read from `decoded[:, 0, :]`, `_train_mpjpe`/`_train_mpjpe_abs` telemetry — is unchanged.

### Routing K into `forward()`

Identical to Design 001. `forward()` signature becomes:

```python
def forward(self,
            feats: Tuple[torch.Tensor, ...],
            k_batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
```

`loss()` and `predict()` both call `self._build_k_batch(batch_data_samples, feats[-1].device)` (when `self.use_k_film`) and pass the result as `k_batch`.

### Invariants preserved

- Output dict keys and shapes unchanged.
- Loss keys unchanged (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`).
- Body-only joint loss (indices 0–21).
- Zero-init guarantee: at step 0, `(gamma, beta) = (0, 0)` → `spatial * 1 + 0 = spatial` → head bit-for-bit baseline.
- Positional encoding is **added before** the FiLM modulation so FiLM can also scale the spatial positional signal; this is intentional and matches the Variant B sketch in `idea.md`.
- No changes to optimizer, LR schedule, data pipeline, batch size, AMP, seed.

---

## 1. `pose3d_transformer_head.py` Changes

Same scaffolding as Design 001 (see Design 001 §1a–§1g). The only differences:

- In the shared `forward()` body, **both** the query-FiLM and spatial-FiLM blocks are present as separate `if` branches gated by `self.k_film_variant`. This keeps the head source unified across all three idea033 designs; the variant string is the only switch.

Explicit forward-body insertion (builder must apply these in this order):

```python
# after input_proj + pos_enc
spatial = spatial + pos_enc

if self.use_k_film and self.k_film_variant == 'spatial':
    if k_batch is None:
        k_batch = torch.zeros(B, 6, device=feat.device, dtype=feat.dtype)
    else:
        k_batch = k_batch.to(device=feat.device, dtype=feat.dtype)
    gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)
    spatial = spatial * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# expand queries (unchanged from baseline)
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

# (query-FiLM block lives here but is skipped in this design because
#  k_film_variant != 'query')
if self.use_k_film and self.k_film_variant == 'query':
    if k_batch is None:
        k_batch = torch.zeros(B, 6, device=feat.device, dtype=feat.dtype)
    else:
        k_batch = k_batch.to(device=feat.device, dtype=feat.dtype)
    gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)
    queries = queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

decoded = self.decoder_layer(queries, spatial)
```

All other changes (imports, `_KFilmMLP`, `__init__` kwargs, `_build_k_batch`, `loss()`/`predict()` routing) exactly match Design 001 §1a–§1g.

---

## 2. `config.py` Changes

In the `head=dict(...)` block, append after `loss_weight_uv=1.0,`:

```python
        # ── Camera-intrinsic FiLM (idea033 / Variant B — spatial-token FiLM) ──
        use_k_film=True,
        k_film_variant='spatial',
        k_film_hidden=64,
```

No other changes to `config.py`.

---

## 3. `pelvis_utils.py` Changes

None.

---

## Expected Behavior After Change

- At step 0, FiLM is identity → head output matches baseline to numerical precision.
- Parameter count added: `~33.8K` (same FiLM MLP as Design 001).
- Keys/values of cross-attention are modulated sample-wise by normalized K before the decoder attends, giving the body-query attention a K-aware feature field to draw from.
- Baseline recoverable via `use_k_film=False`.
- Expected to improve `mpjpe_abs_val` and possibly also body MPJPE (since K-scaled spatial features can help localize K-dependent cues such as limb scale in image plane).

---

## Constraints / Edge Cases

- Same K-missing fallback as Design 001 (baseline guarantees K present in metainfo).
- `pos_enc` is a constant buffer (no grad), so modulating `spatial` after adding it is safe for gradients through FiLM; no double-counting.
- Broadcasting `gamma.unsqueeze(1)` across 960 spatial tokens is cheap: `O(B * 960 * 256)` multiplies, far below backbone cost.
- Must not apply FiLM **before** `input_proj`, since `input_proj` expects raw backbone channels (1024); FiLM output dim is `hidden_dim=256`.
- `k_film_hidden=64` matches Design 001 for controlled comparison across variants.
