## Code Review — idea003/design003

**Verdict: APPROVED**

**Automated check:** PASSED (`python scripts/cli.py review-check-implementation runs/idea003/design003`)

---

### Files Changed

`implementation_summary.md` lists two files changed: `code/pose3d_transformer_head.py` and `code/config.py`. Both are permitted by `design.md`. `pelvis_utils.py` and `train.py` confirmed unmodified (match baseline exactly).

---

### Code Fidelity vs. Design

**`pose3d_transformer_head.py`:**

1. `query_cond_type: str = 'mlp_norm'` added as parameter to `__init__` — CORRECT.
2. `self.query_cond_type` stored — CORRECT.
3. `bottleneck_dim = hidden_dim // 2` (=128) — CORRECT.
4. `self.query_cond_net = nn.Sequential(nn.Linear(hidden_dim, bottleneck_dim), nn.GELU(), nn.Linear(bottleneck_dim, num_joints * hidden_dim))` — CORRECT.
5. `self.query_cond_norm = nn.LayerNorm(hidden_dim)` — CORRECT; `hidden_dim=256`.
6. Init loop over `query_cond_net` with `isinstance(layer, nn.Linear)` guard: `trunc_normal_(std=0.02)` weights, zero biases — CORRECT.
7. `query_cond_norm` NOT explicitly re-initialized — CORRECT; PyTorch default (weight=1, bias=0) is preserved as required.
8. `else: raise ValueError(...)` guard — CORRECT.
9. `_init_head_weights` NOT modified — CORRECT.
10. `forward`: `static_q` broadcast — CORRECT.
11. `global_feat = spatial.mean(dim=1)` after pos_enc — CORRECT.
12. `offsets = self.query_cond_net(global_feat)` — CORRECT.
13. `offsets = offsets.reshape(B, self.num_joints, self.hidden_dim)` — CORRECT.
14. `offsets = self.query_cond_norm(offsets)` applied to reshaped tensor `(B, num_joints, hidden_dim)` — CORRECT; normalises over last dimension (hidden_dim=256), independently per (batch, joint) pair, exactly as specified.
15. `queries = static_q + offsets` — CORRECT.
16. All downstream unchanged — CORRECT.
17. Loss restricted to `_BODY = list(range(0, 22))` — CORRECT.
18. `pelvis_depth`/`pelvis_uv` from `decoded[:, 0, :]` — CORRECT.

Note: The forward method has a stale comment line `# Broadcast joint queries to batch` above the new `# Static joint query embeddings, broadcast to batch` comment (lines 264-265), but the actual code is correct. This is a cosmetic artifact and does not affect correctness.

**`config.py`:**
- `query_cond_type='mlp_norm'` added to `head` dict — CORRECT.
- `output_dir` patched to design003 path — expected.
- All other values identical to baseline — CORRECT.

---

### Test Output

- Test run completed successfully: `[test] Finished.` with no errors.
- 1 epoch of training and validation completed.
- `metrics.csv` produced: `epoch=1, composite_val=458.68, mpjpe_body_val=333.04, mpjpe_pelvis_val=713.75` — finite, plausible for epoch 1.
- Training loss at iter 50: `loss=1.550359` — finite.
- Note: `grad_norm=31.53` at iter 50 is higher than design001/002 (~8-10), consistent with the design note that LayerNorm-normalised offsets have unit-magnitude at init, which can cause slightly larger initial gradients. This is expected behavior per the design and does not indicate a bug.
- GPU memory usage: 10647 MiB — within 1080 Ti limit.

---

### Invariant Check

No invariant files modified. `pelvis_utils.py`, `train.py` match baseline exactly.
