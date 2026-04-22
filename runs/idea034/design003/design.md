# Design 003 — Variant C: Metric 3D PE Injected into Cross-Attention Keys Only (Not Values)

**Design Description:** Same per-token metric 3D unprojection and MLP embedding as design001, but the PE_3D addend is applied **only to the keys** of the cross-attention in the decoder layer — values remain pure appearance (`input_proj(feat) + PE_2D` only). This decouples routing (geometry-based, via keys) from feature aggregation (appearance-based, via values), mirroring the idea021 cross-attn-bias principle with metric geometry instead of a learned bias. Query-self-attention path is untouched. Zero-init of the final MLP Linear guarantees baseline equivalence at step 0.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `_Metric3DPE` MLP (same as design001), `_extract_depth_map`, `_build_K_batch`, **patch the single `_DecoderLayer` to accept a separate `spatial_keys` tensor distinct from `spatial_values`**, route PE_3D to keys only, thread K/depth through `loss()`/`predict()` into `forward()`.
2. `pelvis_utils.py` — add `unproject_grid_to_metric_3d(...)` (identical function as described in design001/section 3).
3. `config.py` — add kwargs `use_metric_pe_3d=True`, `metric_pe_variant='keys_only'`, `metric_pe_mlp_hidden=256`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0`.

All invariant files unchanged.

---

## Algorithm

### 1. Depth and K extraction — identical to design001

Reuse `_extract_depth_map` and `_build_K_batch` as in design001. Same helper specifications.

### 2. `unproject_grid_to_metric_3d` in `pelvis_utils.py` — identical to design001

Same signature and body.

### 3. MLP embedding module — identical to design001

Use `_Metric3DPE(hidden_dim, mlp_hidden=256)` with zero-init final Linear. Same class body as design001 section 4.

### 4. Patch `_DecoderLayer` to separate keys from values

The baseline `_DecoderLayer.forward(queries, spatial_tokens)` uses `spatial_tokens` as both keys and values in the `cross_attn` call. Change the signature and body to allow a distinct keys tensor:

```python
def forward(self, queries: torch.Tensor,
            spatial_values: torch.Tensor,
            spatial_keys: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        queries:        (B, num_queries, embed_dim)
        spatial_values: (B, num_spatial, embed_dim) — values (= appearance + PE_2D)
        spatial_keys:   (B, num_spatial, embed_dim) | None
                        If None, falls back to ``spatial_values`` (baseline).
    """
    if spatial_keys is None:
        spatial_keys = spatial_values

    # Self-attention — unchanged
    q = self.norm1(queries)
    q2 = self.self_attn(q, q, q)[0]
    queries = queries + self.dropout1(q2)

    # Cross-attention — keys != values when spatial_keys is provided
    q = self.norm2(queries)
    q2 = self.cross_attn(q, spatial_keys, spatial_values)[0]
    queries = queries + self.dropout2(q2)

    # FFN — unchanged
    queries = queries + self.ffn(self.norm3(queries))

    return queries
```

Backwards-compatible: the baseline call `self.decoder_layer(queries, spatial)` still works (keys defaults to values → bit-for-bit baseline behaviour when `spatial_keys is None`).

### 5. Head `__init__` changes

Add to `__init__` signature (after `loss_weight_uv`, before `init_cfg`):

```python
use_metric_pe_3d: bool = False,
metric_pe_variant: str = 'keys_only',
metric_pe_mlp_hidden: int = 256,
metric_pe_depth_clamp_min: float = 0.1,
metric_pe_depth_clamp_max: float = 50.0,
```

Inside `__init__`, after existing modules:

```python
self.use_metric_pe_3d = bool(use_metric_pe_3d)
self.metric_pe_variant = str(metric_pe_variant)
self.metric_pe_depth_clamp_min = float(metric_pe_depth_clamp_min)
self.metric_pe_depth_clamp_max = float(metric_pe_depth_clamp_max)
if self.use_metric_pe_3d:
    assert self.metric_pe_variant == 'keys_only', \
        f"design003 requires metric_pe_variant='keys_only', got {self.metric_pe_variant}"
    self.metric_pe_3d = _Metric3DPE(hidden_dim, mlp_hidden=int(metric_pe_mlp_hidden))
```

### 6. `forward()` signature and body

Change signature:

```python
def forward(
    self,
    feats: Tuple[torch.Tensor, ...],
    metric_xyz: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
```

Replace the existing block from `feat = feats[-1]` through `decoded = self.decoder_layer(queries, spatial)` with:

```python
feat = feats[-1]                                          # (B, C, H, W)
B, C, H, W = feat.shape

spatial = feat.flatten(2).transpose(1, 2)                 # (B, H*W, C)
spatial = self.input_proj(spatial)                        # (B, H*W, hidden_dim)
pos_enc = self._get_pos_enc(H, W, feat.device)
spatial_values = spatial + pos_enc                        # values = appearance + PE_2D

if self.use_metric_pe_3d and metric_xyz is not None:
    pe3d = self.metric_pe_3d(metric_xyz.to(spatial_values.dtype))  # (B, H*W, hidden_dim)
    spatial_keys = spatial_values + pe3d                  # keys = values + PE_3D
else:
    spatial_keys = None                                   # fall through to baseline

queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, J, D)

decoded = self.decoder_layer(queries, spatial_values, spatial_keys)  # (B, J, D)
```

Output projections (`joints_out`, `depth_out`, `uv_out`) are unchanged.

At step 0 (zero-init `_Metric3DPE.fc2`), `pe3d ≡ 0`, so `spatial_keys = spatial_values` and the cross-attention is bit-for-bit identical to the baseline call.

### 7. `loss()` and `predict()` changes — identical to design001 section 7

Same metric_xyz construction before `self.forward(...)`:

```python
feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
metric_xyz = None
if self.use_metric_pe_3d:
    depth_grid = self._extract_depth_map(
        batch_data_samples, feat_h, feat_w, feats[-1].device)
    K_batch, crop_hw = self._build_K_batch(
        batch_data_samples, feats[-1].device)
    from pelvis_utils import unproject_grid_to_metric_3d
    metric_xyz = unproject_grid_to_metric_3d(
        depth_grid, K_batch, crop_hw, feat_h, feat_w,
        d_min=self.metric_pe_depth_clamp_min,
        d_max=self.metric_pe_depth_clamp_max,
    )
pred = self.forward(feats, metric_xyz=metric_xyz)
```

All other logic in `loss()` / `predict()` is unchanged.

### 8. `config.py` changes

In the `head=dict(...)` block, add after `loss_weight_uv=1.0,`:

```python
        # ── Metric 3D PE (idea034 / Variant C — keys only) ──
        use_metric_pe_3d=True,
        metric_pe_variant='keys_only',
        metric_pe_mlp_hidden=256,
        metric_pe_depth_clamp_min=0.1,
        metric_pe_depth_clamp_max=50.0,
```

No other changes to `config.py`.

---

## Exact Expected Behaviour

- At step 0, `self.metric_pe_3d.fc2.weight = 0` and `.bias = 0` → `pe3d = 0` → `spatial_keys = spatial_values`, giving an identical cross-attention result to baseline. Losses at step 0 equal baseline to float precision.
- Cross-attention dot product becomes `q · (spatial_values + PE_3D)`, which is `q · spatial_values + q · PE_3D` — a geometry-only bias on the routing scores. Values aggregated through the attention softmax remain pure appearance (`input_proj(feat) + PE_2D`), so the head's feature path is untouched — geometry drives *where* to look, appearance determines *what* gets returned.
- Added parameters: same as design001 (66.8K for the MLP). The decoder-layer patch adds zero parameters.
- Per-step overhead: unchanged from design001 (~1–2 ms). The cross-attention still does one softmax over `H'*W'=960` keys; using distinct keys vs. values does not change the attention cost.
- Output dict keys and shapes unchanged.

---

## Constraints / Invariants the Builder Must Preserve

All invariants from design001 (items 1–13) apply, plus:

14. **Distinct `spatial_keys` path is load-bearing.** `spatial_keys = spatial_values + pe3d`; `spatial_values = input_proj(feat) + pos_enc` (PE_3D is NOT added to `spatial_values`). Do not collapse to a single `spatial + pe3d` variable — that would make this design identical to design001 (Variant A).
15. **`_DecoderLayer.forward` backwards compatibility.** The new `spatial_keys` parameter defaults to `None` and falls through to `spatial_values`; this preserves the existing API for any call site that does not pass keys (baseline call `self.decoder_layer(queries, spatial)` must still work unchanged).
16. **Self-attention path untouched.** Self-attention still runs on `queries` alone; PE_3D does not touch queries in this variant.
17. **No MLP on values.** Only the keys receive `pe3d`. If Builder inserts `pe3d` into `spatial_values`, the design collapses to Variant A.

---

## Edge Cases

Same set as design001 (missing depth/K, variable `img_shape`, AMP cast, persistent_workers=False). Additionally:

- **`metric_xyz is None`** path: `spatial_keys` remains `None`, `_DecoderLayer.forward` sets `spatial_keys = spatial_values`, cross-attention degrades gracefully to baseline. This is the correct behaviour for unit tests or any code path that skips metric PE.
- **`use_metric_pe_3d=False`** path: same as `metric_xyz is None` above — the `if` branch in `forward()` does not run, `spatial_keys` stays `None`, baseline behaviour is preserved exactly. This keeps the design file usable for future on/off ablations.
