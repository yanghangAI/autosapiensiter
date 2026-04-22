# Design 001 — Variant A: Query FiLM from Normalized K

**Design Description:** Inject per-sample camera intrinsics `k=[fx/W_ref, fy/H_ref, cx/W, cy/H, crop_h/H_ref, crop_w/W_ref]` (6-dim) into the decoder via a small 2-layer MLP that produces FiLM `(gamma, beta)` of dim `2*hidden_dim`, applied as `q <- (1+gamma)*q + beta` to all 70 joint queries once, before the single decoder layer's self-attention; the final `gamma/beta` Linear is zero-initialized so that at step 0 gamma=0,beta=0 and the head is bit-for-bit baseline.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add K-FiLM MLP module, route K via `forward()` from `loss()` / `predict()`, apply FiLM to queries.
2. `config.py` — add FiLM-related kwargs to the `head=dict(...)` block.
3. `pelvis_utils.py` — **unchanged**.

All invariant files (`bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `rgbd_pose3d.py`, `train.py`, `tools/train.py`, `infra/*`) are untouched.

---

## Algorithm

### K extraction and normalization (per-batch)

For each sample `i` in the batch, read the 3x3 intrinsic matrix and image shape already exposed in `batch_data_samples[i].metainfo`:

```python
K_np   = np.asarray(ds.metainfo['K'], dtype=np.float32)       # (3, 3)
img_shape = ds.metainfo.get('img_shape', (640, 384))           # (crop_h, crop_w)
fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
ch, cw = int(img_shape[0]), int(img_shape[1])
```

Construct the normalized 6-dim K vector:

```
k_i = [fx / W_ref, fy / H_ref, cx / cw, cy / ch, ch / H_ref, cw / W_ref]
```

where reference scales are `W_ref = 384.0` and `H_ref = 640.0` (crop canonical size, hardcoded in the head as class-level constants matching baseline `img_w`/`img_h`).

Stack per-sample vectors into a `(B, 6)` float32 tensor `k_batch` on the same device as `feats[-1]`.

### FiLM MLP

```
FiLM:  Linear(6, 64) → GELU → Linear(64, 2 * hidden_dim)
```

- Input: `(B, 6)`
- Hidden width: `film_hidden = 64` (fixed in this design).
- Output: `(B, 2 * hidden_dim)`; split along the last dim into `gamma` and `beta`, each `(B, hidden_dim)`.
- **Zero-init policy:** the final `Linear(64, 2*hidden_dim)` has both `weight` and `bias` zero-initialized so that at step 0, `gamma=0, beta=0`.
- First Linear is trunc_normal(std=0.02).

### FiLM application — Variant A (this design)

Before the decoder layer, after constructing the expanded query tensor `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)` of shape `(B, num_joints, hidden_dim)`:

```python
gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)        # each (B, hidden_dim)
gamma = gamma.unsqueeze(1)                                      # (B, 1, hidden_dim)
beta  = beta.unsqueeze(1)                                       # (B, 1, hidden_dim)
queries = queries * (1.0 + gamma) + beta                        # broadcast to (B, num_joints, hidden_dim)
```

The `(1.0 + gamma)` form guarantees identity at gamma=0 (a standard "safe init" pattern for FiLM layers; see e.g. iDisc, Perez et al. FiLM). All 70 joint queries (body + pelvis + hands) receive the same K-FiLM modulation.

Everything downstream (self-attn, cross-attn, FFN, output Linears) is unchanged.

### Routing K into `forward()`

Baseline `forward(feats)` does not receive `batch_data_samples`. Change the signature to:

```python
def forward(self,
            feats: Tuple[torch.Tensor, ...],
            k_batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
```

- When `self.use_k_film` is `False` (baseline behaviour), `k_batch` is ignored and the FiLM path is skipped entirely.
- When `self.use_k_film` is `True` and `k_batch is None`, the head falls back to a zero vector `torch.zeros(B, 6, device=feat.device)` so that forward still works in contexts where no data samples are available (e.g. dummy shape tests). This fallback also produces identity FiLM at step 0.
- `loss()` and `predict()` **both** build `k_batch` from `batch_data_samples` via the extraction snippet above and pass it to `forward(feats, k_batch)`.

A helper method on the head:

```python
def _build_k_batch(self, batch_data_samples, device):
    vals = []
    for ds in batch_data_samples:
        K_np = np.asarray(ds.metainfo['K'], dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        ch, cw = int(img_shape[0]), int(img_shape[1])
        vals.append([
            fx / self._W_REF, fy / self._H_REF,
            cx / float(cw),   cy / float(ch),
            float(ch) / self._H_REF, float(cw) / self._W_REF,
        ])
    return torch.tensor(vals, dtype=torch.float32, device=device)
```

Invoked in both `loss()` and `predict()` immediately before `self.forward(feats, k_batch)`.

### Invariants preserved

- Output dict keys and shapes unchanged: `joints` `(B, 70, 3)`, `pelvis_depth` `(B, 1)`, `pelvis_uv` `(B, 2)`.
- Loss signatures and keys unchanged: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`.
- Body-only joint loss restriction (indices 0–21) preserved.
- `_train_mpjpe` / `_train_mpjpe_abs` telemetry preserved.
- At step 0, zero-init of the final FiLM Linear yields `gamma=0, beta=0` → `queries * 1 + 0 = queries` → head is bit-for-bit baseline. Main losses step-0 equal baseline to numerical precision.
- No changes to optimizer, LR schedule, data pipeline, batch size, AMP, or seed.

---

## 1. `pose3d_transformer_head.py` Changes

### 1a. Imports

At the top of the file, add after the existing `import torch` line:

```python
import numpy as np
from typing import Optional
```

(The file already has `from typing import Dict, List, Tuple`; extend with `Optional`.)

### 1b. New FiLM MLP nested class (module-level, before `Pose3dTransformerHead`)

```python
class _KFilmMLP(nn.Module):
    """FiLM MLP: 6-dim normalized K → (gamma, beta) of dim 2*hidden_dim.

    Zero-init of the output Linear so the module starts as identity
    (gamma=0, beta=0 → FiLM: q * (1+0) + 0 = q).
    """

    def __init__(self, hidden_dim: int, film_hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(6, film_hidden)
        self.fc2 = nn.Linear(film_hidden, 2 * hidden_dim)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.act = nn.GELU()

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(k)))
```

### 1c. New `__init__` kwargs on `Pose3dTransformerHead`

Add to `__init__` signature (keep all existing args and defaults exactly):

```python
use_k_film: bool = False,
k_film_variant: str = 'query',      # 'query' | 'spatial' | 'pelvis' — this design uses 'query'
k_film_hidden: int = 64,
```

Inside `__init__`, after existing module creations, add:

```python
self.use_k_film = bool(use_k_film)
self.k_film_variant = str(k_film_variant)
self._W_REF = 384.0
self._H_REF = 640.0
if self.use_k_film:
    assert self.k_film_variant in ('query', 'spatial', 'pelvis'), \
        f'unknown k_film_variant {self.k_film_variant}'
    self.k_film_mlp = _KFilmMLP(hidden_dim, film_hidden=int(k_film_hidden))
```

Defaults (`use_k_film=False`) reproduce baseline bit-for-bit.

### 1d. `_build_k_batch` helper

Add as an instance method (code block in the Algorithm section above).

### 1e. Modify `forward()` signature and body

Change:

```python
def forward(self, feats: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
```

to

```python
def forward(self,
            feats: Tuple[torch.Tensor, ...],
            k_batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
```

After the existing lines that build `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)` and **before** `decoded = self.decoder_layer(queries, spatial)`, insert:

```python
if self.use_k_film and self.k_film_variant == 'query':
    if k_batch is None:
        k_batch = torch.zeros(B, 6, device=feat.device, dtype=feat.dtype)
    else:
        k_batch = k_batch.to(device=feat.device, dtype=feat.dtype)
    gamma, beta = self.k_film_mlp(k_batch).chunk(2, dim=-1)       # (B, hidden_dim) each
    queries = queries * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
```

Everything else in `forward()` is unchanged.

### 1f. Modify `loss()` and `predict()`

Exactly two edits in each method: build `k_batch` and forward with it.

In `loss()`, replace:

```python
pred = self.forward(feats)
```

with:

```python
k_batch = self._build_k_batch(batch_data_samples, feats[-1].device) \
    if self.use_k_film else None
pred = self.forward(feats, k_batch)
```

In `predict()`, apply the same substitution. No other changes to these methods.

### 1g. Do not change the `@MODELS.register_module()` decorator, class name, or MRO.

---

## 2. `config.py` Changes

In the `head=dict(...)` block (currently lines 147–162 of `baseline/config.py`), add three new keys immediately after `loss_weight_uv=1.0,` and **before** the closing `),`:

```python
        # ── Camera-intrinsic FiLM (idea033 / Variant A — query FiLM) ──
        use_k_film=True,
        k_film_variant='query',
        k_film_hidden=64,
```

No other changes to `config.py`.

---

## 3. `pelvis_utils.py` Changes

None.

---

## Expected Behavior After Change

- At step 0, logits are bit-for-bit identical to baseline (zero-init guarantees `gamma=0, beta=0`).
- Added parameters: one Linear(6,64) + one Linear(64,512) = `6*64+64 + 64*512+512 = 448 + 33344 ≈ 33.8K` params (<0.03% of the backbone).
- Each joint query is modulated identically per-sample by `(1+gamma, beta)` derived from the per-sample normalized K.
- Baseline behaviour recoverable via `use_k_film=False` in config.
- Metric key coverage unchanged; `mpjpe_abs_val` and `mpjpe_pelvis_val` expected to improve if the mechanism is effective.

---

## Constraints / Edge Cases

- If `ds.metainfo['K']` is missing: fall back to a synthetic identity K `[[1,0,0.5*cw],[0,1,0.5*ch],[0,0,1]]` values before normalization — but baseline guarantees K is present (`PackBedlamInputs` already lists `K` in `meta_keys` in config.py lines 173–174). Builder does not need defensive code beyond what is written.
- `img_shape` may be given as a 2-tuple `(H, W)` (baseline convention). Use index `[0]→crop_h`, `[1]→crop_w`. Do not trust tuple length >2.
- `k_batch` must be float32 on the same device/dtype as `feat` before multiplication. The FiLM MLP is registered as a child module so AMP autocast handles it.
- No per-layer repetition: there is only one decoder layer in the baseline, so FiLM is applied once.
- If the user sets `use_k_film=False`, `self.k_film_mlp` is not created; `forward()` skips the FiLM branch regardless of `k_batch`.
- Normalization constants `_W_REF=384.0`, `_H_REF=640.0` are hardcoded class-level floats; they must match the crop size actually produced by `CropPersonRGBD(out_h=640, out_w=384)`.
