# Design 002 — Variant B: Multi-Scale Sinusoidal 3D PE + Zero-Init Linear

**Design Description:** Same per-token metric 3D unprojection pipeline as design001, but the embedding module is a fixed multi-scale sinusoidal 3D basis (per-axis sinusoids at four characteristic scales σ ∈ {0.25 m, 1 m, 4 m, 16 m}) followed by a single zero-initialised `Linear(basis_dim, hidden_dim)` projection. Richer spectral coverage of the 3D coordinate space than the MLP variant; parameter-light (only the projection Linear is learnable); baseline-identical at step 0 via zero-init.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `_SinusoidalMetric3DPE` module, `_extract_depth_map`, `_build_K_batch`, route K and depth through `loss()`/`predict()` into `forward()`, add PE_3D term to spatial tokens after PE_2D.
2. `pelvis_utils.py` — add `unproject_grid_to_metric_3d(...)` (identical function as described in design001; if design001 is implemented first, reuse it; otherwise add it here with the same signature and body).
3. `config.py` — add kwargs `use_metric_pe_3d=True`, `metric_pe_variant='sinusoidal'`, `metric_pe_sigmas=(0.25, 1.0, 4.0, 16.0)`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0`.

All invariant files unchanged (same list as design001).

---

## Algorithm

### 1. Depth and K extraction — identical to design001

Reuse `_extract_depth_map(batch_data_samples, target_h, target_w, device) -> (B, 1, H', W')` and `_build_K_batch(batch_data_samples, device) -> (K_batch (B,3,3), crop_hw (B,2))` exactly as specified in design001 sections 1 and 2.

### 2. `unproject_grid_to_metric_3d` in `pelvis_utils.py` — identical to design001

Same function body (BEDLAM2 sign convention matching `recover_pelvis_3d`, pixel-centre offset, fp32 internal math, NaN/Inf-safe clamp, returns `(B, H'*W', 3)` in metres). See design001 section 3 for the full implementation.

### 3. Sinusoidal 3D PE module — new class in `pose3d_transformer_head.py`

Add at module scope (before `Pose3dTransformerHead`):

```python
class _SinusoidalMetric3DPE(nn.Module):
    """Multi-scale sinusoidal 3D positional encoding.

    For each 3-vector (X, Y, Z) in metres, compute per-axis sin/cos at
    K characteristic scales sigma_k (in metres). Concatenate over axes
    and scales → basis of dim 6K (2 * 3 axes * K scales). Project to
    hidden_dim with a zero-initialised Linear.

    Let omega_k = 2*pi / sigma_k. Then for each axis a in {X, Y, Z}:
        feat_{a,k} = [sin(omega_k * a), cos(omega_k * a)]
    Concatenation order over (axis, scale) is deterministic:
        [sinX_s0, cosX_s0, sinY_s0, cosY_s0, sinZ_s0, cosZ_s0,
         sinX_s1, cosX_s1, ..., sinZ_s{K-1}, cosZ_s{K-1}]

    The final Linear(6K, hidden_dim) has zero weight and zero bias so
    PE_3D = 0 at step 0.
    """

    def __init__(self, hidden_dim: int, sigmas: Tuple[float, ...] = (0.25, 1.0, 4.0, 16.0)):
        super().__init__()
        assert len(sigmas) >= 1
        self._num_scales = len(sigmas)
        # Register sigmas as a non-persistent buffer of omegas for device routing.
        omegas = torch.tensor(
            [2.0 * math.pi / float(s) for s in sigmas], dtype=torch.float32
        )  # (K,)
        self.register_buffer('_omegas', omegas, persistent=False)
        basis_dim = 6 * self._num_scales
        self.proj = nn.Linear(basis_dim, hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (B, N, 3) metric coords in metres. Returns (B, N, hidden_dim)."""
        # Broadcast: (B, N, 3, 1) * (K,) → (B, N, 3, K)
        arg = p.unsqueeze(-1) * self._omegas.view(1, 1, 1, -1)
        s = torch.sin(arg)                       # (B, N, 3, K)
        c = torch.cos(arg)                       # (B, N, 3, K)
        # Interleave sin/cos within each (axis, scale) pair: stack then flatten.
        # Desired order per scale k: [sinX_k, cosX_k, sinY_k, cosY_k, sinZ_k, cosZ_k]
        # → per-scale block of length 6, outer loop over K scales.
        # Reshape via (B, N, K, 3, 2) → (B, N, K*3*2) = (B, N, 6K).
        feat = torch.stack([s, c], dim=-1)       # (B, N, 3, K, 2)
        feat = feat.permute(0, 1, 3, 2, 4)       # (B, N, K, 3, 2)
        feat = feat.reshape(p.shape[0], p.shape[1], 6 * self._num_scales)
        return self.proj(feat)
```

Note: `math` is already imported in `pose3d_transformer_head.py` (used by `_build_2d_sincos_pos_enc`).

### 4. Head `__init__` changes

Add to `__init__` signature (after `loss_weight_uv`, before `init_cfg`):

```python
use_metric_pe_3d: bool = False,
metric_pe_variant: str = 'sinusoidal',   # this design uses 'sinusoidal'
metric_pe_sigmas: Tuple[float, ...] = (0.25, 1.0, 4.0, 16.0),
metric_pe_depth_clamp_min: float = 0.1,
metric_pe_depth_clamp_max: float = 50.0,
```

Inside `__init__`, after existing module construction:

```python
self.use_metric_pe_3d = bool(use_metric_pe_3d)
self.metric_pe_variant = str(metric_pe_variant)
self.metric_pe_depth_clamp_min = float(metric_pe_depth_clamp_min)
self.metric_pe_depth_clamp_max = float(metric_pe_depth_clamp_max)
if self.use_metric_pe_3d:
    assert self.metric_pe_variant == 'sinusoidal', \
        f"design002 requires metric_pe_variant='sinusoidal', got {self.metric_pe_variant}"
    sigmas = tuple(float(s) for s in metric_pe_sigmas)
    self.metric_pe_3d = _SinusoidalMetric3DPE(hidden_dim, sigmas=sigmas)
```

### 5. `forward()` changes — identical to design001

Change signature to:

```python
def forward(
    self,
    feats: Tuple[torch.Tensor, ...],
    metric_xyz: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
```

After `spatial = spatial + pos_enc` and before `queries = ...`, insert:

```python
if self.use_metric_pe_3d and metric_xyz is not None:
    pe3d = self.metric_pe_3d(metric_xyz.to(spatial.dtype))
    spatial = spatial + pe3d
```

No other `forward()` changes.

### 6. `loss()` and `predict()` — identical to design001 section 7

Build `metric_xyz` before `self.forward(...)` in both methods:

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

### 7. `config.py` changes

In the `head=dict(...)` block, add after `loss_weight_uv=1.0,`:

```python
        # ── Metric 3D PE (idea034 / Variant B — sinusoidal) ──
        use_metric_pe_3d=True,
        metric_pe_variant='sinusoidal',
        metric_pe_sigmas=(0.25, 1.0, 4.0, 16.0),
        metric_pe_depth_clamp_min=0.1,
        metric_pe_depth_clamp_max=50.0,
```

Note: `(0.25, 1.0, 4.0, 16.0)` is a literal Python tuple of floats — allowed by MMEngine (no import required). No other changes to `config.py`.

---

## Exact Expected Behaviour

- At step 0, `self.metric_pe_3d.proj.weight = 0` and `.bias = 0` → `pe3d ≡ 0`. Spatial tokens equal baseline output. Step-0 losses identical to baseline to float precision.
- Basis dimensionality: `6 * 4 = 24` features per token.
- Added parameters: `Linear(24, 256) = 24 * 256 + 256 = 6400` (< 0.004% of backbone). Fewer parameters than design001 (MLP variant, ~66.8K).
- Per-step overhead: dominated by the sinusoidal evaluation `(B, 960, 3, 4)` and the linear projection. Total ~1 ms; negligible.
- σ choices give wavelength coverage over [0.25, 16] m, aligning with (a) fine-scale body-joint spacings (~10–30 cm), (b) torso/limb scales (~0.5–1 m), (c) typical subject-to-camera depths (1–8 m), (d) far-scene depths (16 m).
- Output dict keys and shapes unchanged.

---

## Constraints / Invariants the Builder Must Preserve

All twelve invariants from design001 apply, plus:

13. **Scale list must be a tuple of positive floats,** passed by value (not by reference); `_omegas` is computed at `__init__` from `sigmas` and registered as a non-persistent buffer. Do not hardcode a different σ set inside the module without updating `metric_pe_sigmas`.
14. **Sinusoidal arg is unnormalised metres.** Do not rescale `X, Y, Z` before the sinusoid — the σ values are already in metres. The sinusoidal basis is intentionally periodic at scale σ.
15. **Zero-init is on the projection Linear (`self.metric_pe_3d.proj.weight` and `.bias`)** — not on the basis computation. Do not introduce additional learnable params in the basis.
16. **`basis_dim` must equal `6 * len(sigmas)`** (2 trig functions × 3 axes × K scales); if the Builder changes the concat order, they must keep the dimension count exact to match the Linear input shape.

---

## Edge Cases

Same set as design001 (missing depth/K, variable `img_shape`, AMP cast, persistent_workers=False). Additionally:

- **Sinusoidal wrap-around for large `|Y|` or `|Z|`:** outside principal point, `|Y|, |Z|` can reach ~3 m at edges of the crop (for a 1-m-subject at 2-m depth). The finest scale σ=0.25 m will wrap multiple times across that range — this is expected behaviour of a sinusoidal PE and is handled by the coarser scales disambiguating.
- **Pure-zero depth fallback region (missing NPZ):** after clamp to `d_min=0.1`, the unprojected (X,Y,Z) is a well-defined metric coord on a small near-camera plane. Sinusoidal values are bounded in [-1, 1] regardless. No NaN risk.
