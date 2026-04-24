# Setup Handoff

## Files Created or Modified

### New files
| File | Purpose |
|------|---------|
| `infra/constants.py` | Machine paths, joint indices, training invariants, metric name mapping |
| `infra/metrics_csv_hook.py` | Custom MMEngine hook writing metrics.csv and iter_metrics.csv |
| `baseline/config.py` | Standalone MMEngine config (optimizer, LR, data, model, hooks) |
| `baseline/train.py` | Training wrapper calling tools/train.py with auto work_dir |
| `baseline/pose3d_transformer_head.py` | Transformer head with body-only joint loss (modified from original) |
| `baseline/pelvis_utils.py` | Pelvis unprojection (copied from original) |
| `SETUP_SUMMARY.md` | Setup findings and configuration |
| `PROJECT_SUMMARY.md` | Quick-reference project overview |
| `runs/baseline/` | Validated baseline test output |

### Modified files
| File | Change |
|------|--------|
| `.automation.json` | Fully configured for this project |
| `scripts/slurm/slurm_train.sh` | Customized for Unity HPC (1080ti, conda, PYTHONPATH, no time limit) |
| `scripts/slurm/slurm_test.sh` | Test script: 1 epoch, max_seqs=5, 30-min limit |
| `scripts/slurm/submit_train.sh` | Removed --time flag |
| `agents/Orchestrator/prompt.md` | Updated with project vocabulary |
| `agents/Architect/prompt.md` | Updated with project vocabulary |
| `agents/Designer/prompt.md` | Updated with project vocabulary |
| `agents/Builder/prompt.md` | Updated with project vocabulary |
| `agents/Reviewer/prompt.md` | Updated with project vocabulary |
| `agents/Debugger/prompt.md` | Updated with project vocabulary |
| `runs/idea_overview.csv` | Initialized (empty) |
| `results.csv` | Initialized with header |

## Configured Metrics

- **Primary:** `composite_val` = 0.67 * mpjpe_body_val + 0.33 * mpjpe_pelvis_val (lower is better)
- **All fields:** composite_val, mpjpe_body_val, mpjpe_pelvis_val, mpjpe_rel_val, mpjpe_hand_val, mpjpe_abs_val
- **Progress:** epoch (done at >= 20)

## Commands to Start the Research Loop

```bash
# In a NEW Claude Code session:
# 1. Tell it to act as the Orchestrator
Read agents/Orchestrator/prompt.md and act as the Orchestrator.

# 2. Start the loop
Run the full autonomous research loop.
```

## Recommendation: Run a Full Baseline First

Before starting the research loop, run a full 20-epoch baseline training to establish ground-truth metrics for comparison. The Architect agent needs baseline results to propose meaningful improvements.

```bash
# Set up the baseline design directory (already done during validation)
python scripts/cli.py setup-design baseline/ runs/baseline/

# Submit full 20-epoch baseline training
python scripts/cli.py submit-train runs/baseline/code/train.py baseline
```

Monitor with: `squeue -u $USER`
Check results when done: `cat runs/baseline/output/metrics.csv`

The composite_val from this full run becomes the number to beat.

## Unresolved Items

- None. All sanity checks passed. Pipeline validated end-to-end.

## Reminder

Open a **new Claude Code session** before starting the Orchestrator. The Orchestrator needs a fresh context window to work effectively — do not reuse this setup session.
