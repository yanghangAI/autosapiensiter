**Verdict:** APPROVED

**Summary:** Design C extends design001 with a learnable scalar softmax temperature (`nn.Parameter(torch.tensor(1.0))` passed through `F.softplus` with `clamp(min=1e-3)`). Heatmap sigma=2.0 and loss weight=0.5 match design001. Changes stay within the three experimentable files. Design001 already pre-wires the `uv_heatmap_learnable_temp` gate in the `forward()` pseudo-code; design003 simply sets the flag and explicitly activates it.

**Checks:**
- Design Description present.
- Starting point: `baseline/`.
- Files to change: only the three experimentable files.
- Temperature construction, application, clamp semantics, init value (1.0), and AMP interaction (fp32 param, promotion during division) all specified.
- Optimizer handling explicit: single param group preserved (no new no-decay group) — this is called out as an invariant to preserve, which is correct.
- Loss() unchanged — temperature is baked into `_uv_attn`.
- Checkpoint resume behaviour noted (state_dict compatibility).
- Output contract preserved.
- MMEngine config constraint satisfied (literals only).
- Invariants preserved.

Builder can implement without guessing. Approved.
