"""Project-wide constants for the Sapiens RGBD 3D pose automation."""

# ── Paths ────────────────────────────────────────────────────────────────────
SAPIENS_POSE_DIR = '/home/hangyang_umass_edu/MMC/sapiens/pose'
DATA_ROOT = '/work/pi_nwycoff_umass_edu/hang/BEDLAM2subset'
SPLITS_JSON = '/work/pi_nwycoff_umass_edu/hang/auto/splits_rome_tracking.json'
PRETRAINED_CHECKPOINT = (
    '/home/hangyang_umass_edu/MMC/sapiens/pretrain/checkpoints/'
    'sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth'
)
CONDA_ENV = '/work/pi_nwycoff_umass_edu/.conda/envs/hang'

# ── Architecture ─────────────────────────────────────────────────────────────
NUM_JOINTS = 70
EMBED_DIM = 1024
IMG_H = 640
IMG_W = 384

# ── Joint subsets ────────────────────────────────────────────────────────────
BODY_JOINT_INDICES = list(range(0, 22))

# ── Training invariants ──────────────────────────────────────────────────────
NUM_EPOCHS = 20
SEED = 2026
BATCH_SIZE = 4
GRAD_ACCUM = 8

# ── Metric name mapping (MMEngine evaluator key → CSV column) ────────────────
METRIC_NAME_MAP = {
    'composite/val': 'composite_val',
    'mpjpe/body/val': 'mpjpe_body_val',
    'mpjpe/pelvis/val': 'mpjpe_pelvis_val',
    'mpjpe/rel/val': 'mpjpe_rel_val',
    'mpjpe/hand/val': 'mpjpe_hand_val',
    'mpjpe/abs/val': 'mpjpe_abs_val',
}
