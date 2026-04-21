**Idea Name:** Dedicated Pelvis Query with Decoupled Head

**Approach:** Introduce a separate learnable pelvis query that runs through its own decoder pathway (independent from the 70 joint queries), allowing the pelvis depth/UV head to specialise on absolute localisation without sharing representational capacity with body joint regression.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

Results from idea001 (Multi-Layer Decoder with Intermediate Supervision) reveal a consistent pattern: adding decoder capacity improves `mpjpe_body_val` (−3 to −5 mm) but substantially degrades `mpjpe_pelvis_val` (+14 to +19 mm), producing a net *worse* composite score in all three designs.

The root cause lies in the baseline architecture: **pelvis depth and UV are regressed from joint query token 0** (the first joint query hidden state). This conflates two very different tasks:

1. **Joint token 0** must learn root-relative spatial position of the pelvis joint relative to surrounding body joints — a body-structure task.
2. **Absolute pelvis localisation** (depth in metres, UV in pixels) is a camera-geometric task that depends on scale and position in the full image, not on the skeleton structure.

When we stack additional decoder layers (as in idea001), joint query 0 becomes more specialised for the body-structure task through inter-query self-attention, causing it to *lose* the absolute-position signal needed for pelvis depth/UV prediction. This explains the pelvis degradation pattern.

**Proposed fix:** introduce a dedicated 71st learnable query (`pelvis_query`) that participates only in cross-attention with spatial tokens (no self-attention mixing with joint queries). A separate output head reads from this dedicated token. The 70 joint queries continue exactly as before. This decoupling gives each pathway full freedom to specialise.

## Analysis of Baseline Weak Point

- composite_val = 0.67 × body_val + 0.33 × pelvis_val
- At baseline (epoch 10): body=168.4, pelvis=174.8 → composite=170.5
- Pelvis term contributes 0.33 × 174.8 = 57.7 to the composite
- A 10 mm improvement in pelvis alone saves 3.3 composite points — equivalent to a ~5 mm body improvement
- idea001 designs traded ~−5 mm body for +17 mm pelvis, a bad deal under this composite weight

Decoupling the pelvis pathway targets the pelvis term directly without touching body joint decoding.

## Proposed Variations

**Design A — Decoupled pelvis query, shared decoder layer**
Add a single dedicated `pelvis_query` embedding (1 × hidden_dim). During the decoder forward pass, run cross-attention for the pelvis query against spatial tokens using the *same* `_DecoderLayer` weights as the joint queries, but in a separate call so there is no self-attention mixing with joint tokens. Pelvis depth/UV heads read from this token. Joint queries are unchanged.

This is the minimal-change design: tests whether decoupling alone (without self-attention mixing) helps, with zero extra parameters beyond one embedding vector.

**Design B — Decoupled pelvis query, independent decoder layer**
Give the pelvis query its own `_DecoderLayer` (independent weights, cross-attn only — skip self-attn since there is only one pelvis token). This allows the pelvis pathway to develop entirely different attention patterns than the joint pathway, e.g., attending to background depth cues or image-boundary context that body joints do not need.

**Design C — Decoupled pelvis query + depth feature fusion**
Build on Design B and additionally fuse a lightweight depth-context signal into the spatial tokens *before* pelvis cross-attention. Concretely, concatenate a scalar depth-map summary (mean-pooled over the spatial grid per channel, projected to hidden_dim) as an additional "global depth token" prepended to the sequence. The pelvis cross-attention layer can then attend to this global token to anchor absolute scale. Joint queries use the original sequence without this extra token, keeping their pathway unchanged.

This targets the known difficulty of inferring absolute depth from RGB features alone.

## Implementation Scope

Changes are confined to `pose3d_transformer_head.py`:

- Add `pelvis_query: nn.Embedding(1, hidden_dim)`.
- **Design A**: in `forward()`, call `self.decoder_layer(pelvis_q, spatial)` separately after the joint decoder call; route output to `depth_out` / `uv_out` instead of `decoded[:, 0, :]`.
- **Design B**: add `self.pelvis_decoder = _DecoderLayer(hidden_dim, num_heads, dropout)` with cross-attn-only path (self-attn noop for single-token case); call it for pelvis_query only.
- **Design C**: add `self.depth_proj: nn.Linear(hidden_dim, hidden_dim)` for global depth token; prepend to spatial before pelvis decoder; no change to joint path.
- `config.py`: expose `decouple_pelvis: True` flag and `pelvis_decoder_type: 'shared'|'independent'|'depth_fused'` as head kwargs.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, or data pipeline.

## Expected Outcome

- **Primary gain**: pelvis accuracy should improve significantly because the dedicated query is free to attend to scale-relevant spatial regions without being constrained by body joint self-attention dynamics.
- **Secondary effect**: body joint accuracy should be unaffected or marginally improved because query 0 no longer needs to serve the dual purpose of body-joint and absolute-pelvis regression.
- **Net composite**: targeting −10 to −20 mm pelvis improvement (−3 to −7 composite points) with neutral body impact.

## Risk and Mitigation

- **Memory**: adding one extra query and optionally one extra decoder layer is negligible on the 1080 Ti (< 5 MB). No OOM risk.
- **Single-query self-attention**: for Design B/C, `nn.MultiheadAttention` with a single query token collapses self-attention to a no-op (Q=K=V → identity after softmax). Safest mitigation: skip `self_attn` call entirely for pelvis decoder, or just let it run (the result is numerically harmless).
- **Depth feature availability**: Design C assumes the backbone feature map `feats[-1]` encodes meaningful depth signal. With RGBD input, the backbone processes concatenated RGB+D, so depth information is present in the feature map by construction — this assumption is valid.
- **Interaction with idea001 learnings**: if multi-layer decoding for joints is later desired, it can be layered on top of this decoupling without conflict.
