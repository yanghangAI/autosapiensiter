**Idea Name:** Content-Adaptive Query Initialization

**Approach:** Replace purely static joint-query embeddings with image-conditioned queries by adding a lightweight MLP that maps globally-pooled spatial features to per-joint embedding offsets, giving the single decoder layer a warm-start tailored to each image rather than a fixed random initialization.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder uses a single `_DecoderLayer` and 70 purely static joint queries (learned `nn.Embedding`, initialized from trunc_normal). Every image begins decoding from the same fixed query embedding regardless of subject size, pose diversity, or depth. With only one decoder layer, the cross-attention must both *locate* the correct spatial region and *refine* the joint coordinate in a single pass — a harder task when the queries carry no image-level context.

### Evidence from idea001

idea001 stacked 2–4 decoder layers to give more processing steps. Results at epoch 10:
- Body improved (−3 to −9 mm) but pelvis degraded (+14 to +19 mm).
- The composite score was *worse* than baseline in all three designs.

The lesson is that simply adding capacity (more layers, more self-attention) does not help and can hurt. The single-layer baseline appears to be a reasonably efficient architecture; the bottleneck is not depth of decoding but rather the *quality of the query starting point*.

### Evidence from idea002

idea002 (Dedicated Pelvis Query) is a structural fix for the pelvis pathway. It does not address the body query initialization problem. Body MPJPE remains ~168 mm even after this fix (based on design).

### Why content-adaptive initialization should help

Static queries force the single cross-attention layer to simultaneously (a) determine *where* to attend in the spatial token sequence and (b) aggregate the attended information into a joint coordinate. If the query already encodes a coarse image-level signal — for example, a global body scale hint or rough body-frame orientation — the cross-attention can spend its capacity on fine-grained localization rather than coarse search.

In transformer detection (Conditional DETR, DAB-DETR, Anchor DETR), conditioning queries on global context consistently reduces the number of required decoder layers and improves convergence speed. In our 20-epoch budget, faster convergence directly translates to better epoch-10 and epoch-20 scores.

The change is confined to `pose3d_transformer_head.py`:
- Add a small MLP `query_cond_net: Linear(hidden_dim, num_joints * hidden_dim)` (or a two-layer MLP with bottleneck) that takes globally-averaged spatial tokens as input and outputs per-joint additive offsets.
- At each forward pass: `queries = static_queries + query_cond_net(spatial.mean(1)).reshape(B, num_joints, hidden_dim)`.
- The rest of the decoder (cross-attention, FFN, output projections) is unchanged.

This is strictly additive: at initialization the MLP outputs near-zero, so the model starts close to the baseline. Gradient flows through the MLP naturally during training.

---

## Proposed Variations

**Design A — Single-linear global conditioning (minimal)**

A single `nn.Linear(hidden_dim, num_joints * hidden_dim)` applied to the mean-pooled spatial tokens. Output is reshaped and added to the static query embeddings. This is the fewest-parameter version: one weight matrix of shape `(hidden_dim, num_joints * hidden_dim) = (256, 70*256) ≈ 4.6 M params`. This tests whether any global image signal is useful at all.

Parameter overhead: ~4.6 M additional params (5% of head params). No extra attention operations. Minimal memory impact.

**Design B — Two-layer MLP global conditioning with bottleneck**

Replace the single linear with a bottleneck MLP:
`Linear(hidden_dim, hidden_dim // 2) → GELU → Linear(hidden_dim // 2, num_joints * hidden_dim)`
Bottleneck dimension 128. This reduces params to 256*128 + 128*(70*256) ≈ 2.4 M while adding a nonlinearity that lets the network compose global features before projecting to query space. The bottleneck also acts as a compression that encourages the MLP to learn a compact scene representation (e.g., body scale, global orientation) rather than per-query noise.

**Design C — Two-layer MLP global conditioning + LayerNorm on offset**

Same architecture as Design B, but apply a `nn.LayerNorm(hidden_dim)` to the per-joint offsets before adding them to the static queries. This prevents the dynamic offsets from dominating the static component during early training (a common training instability in additive conditioning). Effectively the offset magnitude is normalized, and only its direction matters — the static query still controls the offset scale. This is the most stable variant and tests whether normalization of the adaptive component matters.

---

## Implementation Scope

All changes are in `pose3d_transformer_head.py`:

1. In `__init__`:
   - Add `self.query_cond_net` (Design A: `nn.Linear`; Design B/C: `nn.Sequential`).
   - Design C only: add `self.query_cond_norm = nn.LayerNorm(hidden_dim)`.
   - Initialize MLP weights with trunc_normal std=0.02 and zero biases so offsets start near zero.

2. In `forward()`:
   - Compute `global_feat = spatial.mean(dim=1)` (B, hidden_dim) — after input_proj and positional encoding.
   - Compute `offsets = self.query_cond_net(global_feat).reshape(B, num_joints, hidden_dim)`.
   - Design C: apply `self.query_cond_norm` per-joint before adding.
   - `queries = static_queries + offsets`.
   - Pass conditioned queries to `self.decoder_layer` as before.

3. In `config.py`:
   - Expose `query_cond_type: 'linear' | 'mlp' | 'mlp_norm'` as a head kwarg.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, or training infrastructure.

---

## Expected Outcome

- **Primary gain**: improved body MPJPE through better cross-attention initialization, targeting −10 to −20 mm body improvement (−7 to −13 composite points) relative to baseline.
- **Pelvis**: unchanged — pelvis depth/UV still reads from joint query 0 as in the baseline. Expected neutral to slight improvement (global scale context may marginally help depth regression).
- **Convergence**: the adaptive initialization may accelerate early learning. Expect clearer separation between designs at epoch 10.
- **Composite target**: aim for composite_val < 160 (vs. baseline 170.5).

---

## Risk and Mitigation

- **Memory**: no extra attention operations. One extra MLP forward per step, negligible cost on 1080 Ti.
- **Training instability from large offsets**: mitigated by zero-bias initialization and (in Design C) LayerNorm on offsets. If offsets grow too large the static query signal is overwhelmed, but this is easily caught by watching loss curves.
- **Interaction with idea002**: the dedicated pelvis query from idea002 would not use `query_cond_net` (it has its own 1-token pathway). If both ideas eventually combine, the joint query conditioning and pelvis decoupling are orthogonal and composable.
- **Spatial mean-pooling loses structure**: global average-pool discards spatial layout. For a single human body centred in a crop this is acceptable — body scale and rough depth are well-captured by global statistics. A future extension could use spatial-attention pooling, but that adds complexity better left to a later idea.
