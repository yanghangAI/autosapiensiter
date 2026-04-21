# Code Review Log — idea019/design003

## Entry 1 — 2026-04-21

**Verdict: APPROVED**

review-check-implementation passed. Head file identical to design001/002 (correct — all design003 differences controlled by config kwargs). Config has `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4` — all literal values matching design spec. Two independent `_DeformableDecoderLayer` instances in `decoder_layers` ModuleList. `has_intermediate_sup=True` activates `intermediate_joints_out = nn.ModuleList([nn.Linear(256,3)])` and `loss/joints_inter0/train` (weight 0.4) in loss(). All five loss terms appeared in training log. Test ran to completion with finite losses, no errors, no OOM (8643 MB).
