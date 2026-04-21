**Idea Name:** Iterative Pose Refinement via Coordinate-Conditioned Decoder

**Approach:** After the decoder produces an initial 3D joint prediction, encode those predicted coordinates with a small positional encoding MLP and add them back to the queries for a second decoder pass, so the second pass has explicit spatial knowledge of where the first pass placed each joint and can produce residual corrections conditioned on that estimate.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder is a **one-shot** predictor: a single decoder layer maps static joint query embeddings + spatial tokens to 3D coordinates in one forward pass. There is no mechanism for the model to re-examine the image *given an initial pose estimate* and fix systematic errors. In DETR-style detection, this one-shot paradigm was superseded by **iterative refinement** (Deformable-DETR, DAB-DETR, DN-DETR), where each decoder layer takes the *previous layer's coordinate output as an explicit input* (not just its hidden state) and predicts a residual correction. Iterative refinement consistently improves both localisation accuracy and convergence speed under fixed-epoch budgets.

### Why this is different from idea001 (multi-layer decoder)

idea001 stacked multiple decoder layers. Each added layer operates only on hidden-state representations — there is no explicit coordinate feedback. The head still reads only the *final* layer's hidden state to produce coordinates. In practice, idea001 improved body MPJPE by 5–9 mm but **regressed pelvis MPJPE by +14 to +19 mm**, likely because the extra self-attention over all 70 queries over-specialised token 0 away from the absolute-position task.

By contrast, iterative refinement:
1. Reads coordinate predictions after each decoder pass and feeds them back as an additional *input* to the next pass.
2. Lets the second pass be a **residual** predictor — it learns to add small corrections on top of the first pass, rather than re-regressing from scratch. Residual prediction is known to train faster and to be more stable (He et al. 2016; the same logic that makes ResNet work).
3. Anchors every joint query to an explicit 3D coordinate estimate, giving the decoder access to a spatial prior that is *data-dependent* rather than fixed by the initial query embedding.

### Why this is different from idea003 (content-adaptive query initialization)

idea003 conditions the initial queries on a single globally-pooled feature vector. This gives the model an image-dependent warm start *once*, before the decoder runs. It does **not** re-condition based on predicted coordinates. Iterative refinement is the analogous idea in the *output space*: condition subsequent passes on *what the model just predicted*.

### Why this is different from idea008 (body-focused decoder) and idea002 (decoupled pelvis)

Those ideas change **what** the decoder processes (22 queries, a separate pelvis query). Iterative refinement changes **how many passes** the decoder runs and what they are conditioned on. The two axes are orthogonal: the body-focused decoder of idea008 can itself be made iterative, and so can idea002's decoupled pelvis pathway.

### Grounding in results

- Baseline composite 168.67 → best seen composite 154.85 (idea002/design003). There is still a large gap to close.
- mpjpe_abs is the worst metric (baseline 454, best 320). Absolute pose accuracy is highly sensitive to *small systematic errors* in both joint XYZ and pelvis depth. Iterative refinement is specifically good at fixing systematic errors: a second pass can observe "pass 1 put the left wrist at (0.3, 0.2, 0.5) but the image feature at that spatial location is weaker than at (0.35, 0.2, 0.5)" and shift the prediction accordingly.
- Body MPJPE across all prior ideas stayed in 140–166 mm range. Explicit coordinate feedback is a new signal; it has a reasonable chance of breaking the 140 mm floor.

## Analysis of Baseline Weak Point

The baseline's single decoder pass has three properties that limit accuracy:

1. **No self-correction**: once the decoder emits a prediction, there is no mechanism to revisit the image given that prediction and check for consistency.
2. **Coordinate-free queries**: the joint queries are static learnable embeddings with no spatial interpretation. They must simultaneously encode "which joint am I?" and "roughly where is this joint?" from scratch every forward. The only way the query "knows" where the joint is is through gradients during training; at inference, the query has no explicit spatial anchor.
3. **Absolute coordinate regression is hard**: regressing 3D root-relative coordinates in metres from transformer hidden states is a harder optimisation problem than regressing a residual on top of an existing estimate. A two-pass residual setup breaks this into (a) "produce an approximate pose" and (b) "refine it locally", a classic divide-and-conquer that regression networks benefit from (cf. cascaded hourglass networks for 2D pose).

Iterative refinement addresses all three.

## Proposed Variations

**Design A — Two-pass coordinate-conditioned decoder (minimal)**

Run the decoder twice with the *same* weights. After the first pass:
1. Read initial joint coordinates `joints_1 = joints_out(decoded_1)` → `(B, 70, 3)`.
2. Encode these coordinates with a small MLP `coord_enc: Linear(3, hidden_dim) → GELU → Linear(hidden_dim, hidden_dim)`, giving `(B, 70, hidden_dim)`.
3. Compute refined query inputs: `queries_2 = decoded_1 + coord_enc(joints_1)`.
4. Run decoder second pass: `decoded_2 = decoder_layer(queries_2, spatial)`.
5. Read residual coordinates: `joints_residual = joints_out(decoded_2)` → `(B, 70, 3)`.
6. Final output: `joints_final = joints_1 + joints_residual`.

The pelvis depth/UV are read from `decoded_2[:, 0, :]` so they also benefit from the refinement pass.

`coord_enc` is initialised so that its *output magnitude is near zero* at start (zero-init the last linear layer's weights, as in Deformable-DETR). This ensures the refined queries match the first pass's hidden states at init, so the network begins training in an approximately baseline-equivalent state and gradually learns to use coordinate feedback.

**Design B — Two-pass with intermediate supervision and shared weights**

Same as Design A, but add a supervised joint loss on `joints_1` with weight 0.5 (in addition to the full-weight loss on `joints_final`). This is direct supervision of the first pass, encouraging it to converge to a reasonable pose estimate before the refinement step acts. In DETR-style iterative refinement, intermediate supervision is standard practice and prevents the refinement head from having to also drive early training.

The pelvis supervision is applied only on the final (pass-2) outputs, since the pelvis token's information mixes across passes and the Design A coordinate encoding does not apply to depth/UV.

**Design C — Two-pass with independent decoder layer weights (full capacity)**

Give pass 2 its own set of decoder layer weights (`decoder_layer_2`), rather than sharing with pass 1. Keep the intermediate supervision from Design B. This allows the second pass to be structurally different from the first: in principle, pass 1 learns the "rough pose" task and pass 2 learns the "local refinement" task with specialised attention patterns (e.g., smaller receptive field, higher-frequency cross-attention).

Parameter cost: one extra `_DecoderLayer` (~1.3M params for hidden_dim=256) — well within 1080 Ti budget.

## Implementation Scope

All changes confined to `pose3d_transformer_head.py` and `config.py`:

**`pose3d_transformer_head.py`:**
1. `__init__`: Add `self.coord_enc = nn.Sequential(Linear(3, hidden_dim), GELU(), Linear(hidden_dim, hidden_dim))` with zero-init on the last linear layer's weights. Accept `num_refine_passes: int = 2`, `shared_decoder: bool = True`, `intermediate_supervision_weight: float = 0.0` kwargs.
2. `__init__` (Design C): Add `self.decoder_layer_2 = _DecoderLayer(...)` if `shared_decoder=False`.
3. `forward()`: Run decoder pass 1, compute coordinates, compute coord_enc, add to decoded, run decoder pass 2 (same or different weights). Read pelvis depth/UV from pass-2 token 0. Return `{'joints': joints_final, 'joints_initial': joints_1, 'pelvis_depth', 'pelvis_uv'}`.
4. `loss()`: Compute the standard body-joint loss on `pred['joints']`. If `intermediate_supervision_weight > 0`, add `intermediate_supervision_weight * loss_joints_module(pred['joints_initial'][:, _BODY], gt_joints[:, _BODY])` as `loss/joints_init/train`.
5. `predict()`: Use `pred['joints']` as usual; `joints_initial` is training-only.

**`config.py`:**
- Add `num_refine_passes=2`, `shared_decoder=True|False`, `intermediate_supervision_weight=0.0|0.5` to head kwargs as integer/bool/float literals.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

## Expected Outcome

- **Primary gain — body MPJPE**: residual refinement is well-suited for the 70-joint regression task; the second pass sees both spatial features AND the initial coordinate estimate, enabling local corrections. Target: `mpjpe_body_val < 140` (best prior 140.96 — idea002/design002).
- **Secondary gain — mpjpe_abs**: because the pelvis token is read from the refined pass, it benefits from the image re-examination step. Target: `mpjpe_abs < 400` (baseline 455, best prior 320 — idea008/design003).
- **Pelvis MPJPE**: expected neutral; the refinement does not directly address pelvis regression (this is complementary to idea002/idea010). If pelvis token 0 is a problem (as idea001 showed), Design C with independent decoder layer weights may help because the second pass's self-attention can re-specialise token 0 for absolute regression.
- **Composite target**: aim for `composite_val < 153`, improving on best prior (154.85) by capturing a new axis of improvement.

## Risk and Mitigation

- **Zero-init of coord_enc**: must be strict. If the final linear in `coord_enc` has nonzero bias or is default-initialised, the refined queries will be far from the baseline at init, potentially destabilising training. Mitigation: explicitly `nn.init.zeros_` the last linear's weight AND bias in `_init_head_weights`. This is a standard Deformable-DETR technique.
- **Gradient flow through `joints_1`**: the residual output `joints_final = joints_1 + joints_residual` means `joints_1` also receives gradient from the final loss via `coord_enc`. This is intended: the first pass learns to produce a good estimate *because* the final loss flows through it. With intermediate supervision (Design B/C), the first pass also has a direct supervision signal.
- **Memory / speed**: one extra decoder forward pass (~1.3M params worth of compute); negligible on 1080 Ti. Per-batch wall time increases by ≤30%; still fits within the 20-epoch budget on 1080 Ti.
- **Overfitting to pass-1 errors**: if pass 1 systematically over-predicts (e.g. biases limbs toward the root), the residual pass might just learn to invert that bias. Mitigation: with intermediate supervision (Design B/C), pass 1 is also supervised to be accurate, preventing it from becoming a mere "noise generator" that pass 2 cleans up.
- **Interaction with prior ideas**: orthogonal to idea002 (decoupled pelvis), idea005 (loss weighting), idea006 (attention bias), idea008 (body-focused queries), idea009 (spatial dropout), idea010 (reprojection loss). Can compose with any of them. Conceptually overlaps slightly with idea001 (multi-layer decoder) but is distinct: idea001 is multi-layer-of-hidden-states, this is two-pass-with-coordinate-feedback. The residual-correction prior is absent in idea001.
- **MMEngine config constraint**: `num_refine_passes`, `shared_decoder`, `intermediate_supervision_weight` are integer/bool/float literals. No imports required.
- **Eval/inference compatibility**: `predict()` returns the refined `joints` tensor as before; downstream metric sees the same `(B, 70, 3)` shape. `joints_initial` is training-only and not written to predictions.
