# Code Review Log — idea006 / design003

## Entry: 2026-04-16

**Verdict: APPROVED**

All implementation changes match the design spec exactly. `_DecoderLayer.__init__` accepts `attn_bias_mode` with correct dispatch: `'per_head'` registers `nn.Parameter(torch.zeros(num_heads,num_joints,num_joints))`, `'shared'` registers `(num_joints,num_joints)`, `'none'` sets `None`. `self.num_heads` and `self.attn_bias_mode` stored as attributes. `forward` reads `B=queries.shape[0]`, expands via `.unsqueeze(0).expand(B,-1,-1,-1).contiguous().reshape(B*num_heads,J,J)`, passes as `attn_mask`. `Pose3dTransformerHead` maps `attn_bias_type` → `attn_bias_mode` correctly. `config.py` adds `attn_bias_type='per_head'` as string literal. Invariant files unmodified. Test run completed cleanly with valid metric output.
