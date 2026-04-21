**Idea Name:** Body-Focused Decoder with Lightweight Hand Upsampling

**Approach:** Replace the 70-query decoder with a 22-query body-only decoder (joints 0–21 plus pelvis token), then recover hand joint predictions (indices 22–69) via a shared linear projection from body query outputs, so that self-attention and cross-attention are no longer polluted by 48 unevaluated hand queries.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder runs self-attention and cross-attention over all 70 joint queries simultaneously. Of these, 48 queries (indices 22–69) correspond to hand joints that are **not evaluated by the composite metric** (`composite_val = 0.67 * mpjpe_body_val + 0.33 * mpjpe_pelvis_val`), and hand MPJPE does not factor into any reported optimization target.

This creates two concrete problems:

### 1. Self-attention pollution

The self-attention in `_DecoderLayer` is `nn.MultiheadAttention` over all 70 queries. During self-attention, every body joint query attends to all 48 hand joint queries (and vice versa). The attention matrix is 70×70 = 4,900 elements. The hand queries carry strong high-frequency texture-level signals (finger articulation) that are structurally very different from the coarse geometric structure of torso/limb joints. Since the attention is unconstrained, body joint queries can "borrow" information from hand queries during self-attention, injecting irrelevant spatial context. Evidence from **idea006** and the observed pattern in **idea001** suggests this cross-query contamination is real: multi-layer decoder that should strengthen body structure also degrades the pelvis token, likely because the expanded self-attention across all 70 queries overwhelms the pelvis token with body/hand signals.

### 2. Wasted decoder capacity

With 70 queries, the cross-attention matrix is (70, H'×W') = (70, 960). Of these, 48 rows correspond to hands. Each hand query cross-attends to all 960 spatial tokens, consuming ~69% of the cross-attention compute and capacity. Since hand joints are never backpropagated for body-relevant gradients, this is decoder capacity spent on unevaluated targets.

### What this idea proposes

Run the full transformer decoder pipeline over only **22 body queries** (indices 0–21):
- Self-attention is now 22×22 = 484 elements (90% reduction vs. 4,900).
- Cross-attention is (22, 960) — 69% fewer rows than baseline.
- The body decoder operates in a "clean" semantic space: all 22 queries correspond to evaluated body joints, so gradients from the joint loss flow to all active queries.

After decoding, hand joint predictions (indices 22–69) are recovered by a **linear upsampling** from the 22 decoded body query features:
```
hand_preds (B, 48, 3) = body_features (B, 22, hidden_dim) @ W_hand (22, 3*48).reshape(B, 48, 3)
```
This is parameter-efficient (22 × 3 × 48 = 3,168 scalars) and ensures hand outputs still exist in the output tensor for bookkeeping. Since hand joints do not contribute to any reported metric, the hand upsampling is effectively a dummy decoder trained only for self-consistency.

Pelvis-specific tokens (depth, UV) are still regressed from token 0 of the body query set, as in the baseline.

### Evidence grounding

- **idea001** (multi-layer decoder): best design achieved composite_val 162.00 at epoch 13, a 4.7% improvement. Body MPJPE improved but pelvis MPJPE regressed. A body-focused decoder could avoid this regression by keeping the attention computation "clean."
- **idea006** (self-attention bias): attempts to sculpt the 70×70 attention matrix with an additive bias. The body-focused decoder is a harder structural prior — instead of biasing toward body structure, it removes non-body queries entirely.
- **idea007** (cross-attention routing): routes each joint group to its spatial region. The body-focused decoder removes hand-query cross-attention rows entirely rather than redirecting them.

---

## Proposed Variations

### Design A — Body-only decoder, hand outputs discarded (diagnostic)

Run the decoder on 22 body queries only. **Do not output hand joints at all** — truncate the joints output to shape `(B, 22, 3)` and zero-pad to `(B, 70, 3)` after detach (padding is never used for loss or metric). This is the cleanest diagnostic: any performance change is attributable purely to removing hand query contamination.

Changes:
- `Pose3dTransformerHead.__init__`: Change `self.joint_queries = nn.Embedding(22, hidden_dim)` (only body queries).
- `forward()`: decoder runs on 22 queries; `joints_out` produces `(B, 22, 3)`. Zero-pad output to `(B, 70, 3)`.
- `loss()`: Joint loss restricted to `_BODY = list(range(0, 22))` — same as baseline, no change needed.
- `config.py`: Add `num_body_queries: 22` as head kwarg.

### Design B — Body-only decoder with linear hand recovery

Same 22-query decoder as Design A, but recover hand predictions via a linear projection rather than zero-padding:
- Add `self.hand_proj = nn.Linear(22 * hidden_dim, 48 * 3)` to the head.
- In `forward()`: after decoding, flatten body features `(B, 22, hidden_dim)` → `(B, 22*hidden_dim)`, project to `(B, 48*3)`, reshape to `(B, 48, 3)`, and concatenate with body joints output: `joints = torch.cat([body_joints, hand_joints], dim=1)` → `(B, 70, 3)`.
- Add a small auxiliary hand loss (e.g. weight 0.1) so the projection learns meaningful geometry and does not produce NaN. The hand loss does not contribute to composite metric but provides useful gradient signal to keep the body queries anchored in a pose-space.
- Loss change: add `losses['loss/hand_aux/train'] = 0.1 * self.loss_joints_module(pred_joints[:, 22:], gt_joints[:, 22:])`.

This design tests whether having hand-consistent output (without hand-query contamination in the decoder) improves convergence compared to both baseline and Design A.

### Design C — Body-only decoder with two-layer MLP hand recovery + aux loss upweighted

Same as Design B, but replace the single `hand_proj` linear with a 2-layer MLP:
```
hand_proj: Linear(22*hidden_dim, hidden_dim) → GELU → Linear(hidden_dim, 48*3)
```
And increase the hand auxiliary loss weight to 0.3. The motivation: a single linear may not capture the nonlinear relationship between body pose and hand pose implied by the skeleton. A small bottleneck MLP can model this better and provide richer gradient to the body decoder via the hand auxiliary signal.

Parameter cost: 22×256×256 + 256×144 ≈ 1.47M parameters — within budget for a 1080 Ti.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`:

**`pose3d_transformer_head.py`:**
1. `__init__`: Accept `num_body_queries: int = 22` as constructor kwarg. Change `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)`.
2. `__init__` (Design B/C): Add `self.hand_proj` module.
3. `forward()`: Decode over `num_body_queries` queries. Assemble full 70-joint output by concatenation or zero-padding.
4. `loss()`: No change to body joint loss indices (already `range(0, 22)`). Add hand aux loss for Design B/C.

**`config.py`:**
- Add `num_body_queries=22` to head kwargs (Design A/B/C).
- Optionally add `hand_aux_loss_weight=0.1` or `0.3` (Design B/C).

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

---

## Expected Outcome

- **Primary gain**: removing 48 hand queries from self-attention reduces attention pollution by ~70% in the query domain; body joint queries attend only to each other and to the pelvis token. This should improve body MPJPE by eliminating hand-induced interference.
- **Pelvis**: expected to maintain or improve (pelvis token 0 is now in a 22-query self-attention set where it only needs to negotiate with body joints, not hands). This directly addresses the pattern seen in idea001 where pelvis degraded with more decoder capacity.
- **Design A**: diagnostic — does removing hand queries alone improve body MPJPE? Expected −8 to −15 mm.
- **Design B**: adds hand consistency via linear recovery. Auxiliary gradient from hand loss may improve convergence of body decoder. Expected −10 to −18 mm body MPJPE.
- **Design C**: richest variant; MLP hand recovery with stronger aux signal. Highest potential if the MLP provides useful gradient regularisation to the body decoder.
- **Composite target**: aim for composite_val < 158 (vs. baseline 169.75, idea001 best = 162.00 at epoch 13).

---

## Risk and Mitigation

- **Hand auxiliary loss instability**: the hand MLP in Design B/C is trained with a small weight (0.1–0.3) and may produce noisy gradients if hand GT is unreliable. Mitigation: use `SoftWeightSmoothL1Loss` (same as body) with its built-in robustness to outliers.
- **Output tensor shape compatibility**: downstream metric code (`bedlam_metric.py`) likely expects `(B, 70, 3)` joint outputs. Zero-padding (Design A) or concatenation (Design B/C) both produce the correct shape. Designer should verify the metric ignores indices 22–69 for body MPJPE computation.
- **Zero-pad gradient block**: in Design A, the zero-padded hand region has no gradient. This is intentional — the hand queries do not exist. The Designer should use `torch.zeros` with `requires_grad=False` for the padding.
- **Memory**: smaller query set (22 vs. 70) reduces self-attention memory quadratically. Cross-attention rows reduced 69%. The hand MLP in Design C adds ~1.47M parameters but no extra attention computation. Net result: lower memory usage than baseline for the decoder portion.
- **MMEngine config constraint**: `num_body_queries` is an integer literal. `hand_aux_loss_weight` is a float literal. No imports required. Fully compliant.
- **Interaction with other ideas**: this idea is orthogonal to idea005 (loss weighting), idea006 (self-attention bias), and idea007 (cross-attention routing). It can compose with any of them. It is conceptually complementary to idea001 (multi-layer decoder): a 2-layer decoder over 22 body queries would be even more powerful than either alone.
