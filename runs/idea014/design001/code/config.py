# Sapiens 0.3B RGBD 3D Pose Estimation — BEDLAM2 Baseline Config
#
# MMEngine config — NO Python import statements allowed.
# Use __import__() for stdlib calls and hardcode all values as literals.

_base_ = []

# ── Two-stage training controlled by STAGE env var ───────────────────────────
# STAGE=1 → train100.txt, 20 epochs    (default)
# STAGE=2 → train400.txt, 10 epochs    (only run when stage-1 beats baseline)
_stage = int(__import__('os').environ.get('STAGE', '1'))

_repo_root = '/work/pi_nwycoff_umass_edu/hang/autosapiens_iter'
_read_lines = lambda p: [
    (l.strip()[:-4] + '.npz' if l.strip().endswith('.npy') else l.strip())
    for l in open(p).read().splitlines() if l.strip()
]
_train_file = 'train100.txt' if _stage == 1 else 'train400.txt'
_splits = dict(
    train=_read_lines(_repo_root + '/' + _train_file),
    val=_read_lines(_repo_root + '/val200.txt'),
)

# ── Custom imports (register modules with MMEngine) ──────────────────────────
custom_imports = dict(
    imports=[
        'mmpose.models.pose_estimators.rgbd_pose3d',
        'mmpose.models.backbones.sapiens_rgbd',
        'pose3d_transformer_head',
        'mmpose.models.data_preprocessors.rgbd_data_preprocessor',
        'mmpose.datasets.datasets.body3d.bedlam2_dataset',
        'mmpose.datasets.transforms.bedlam2_transforms',
        'mmpose.evaluation.metrics.bedlam_metric',
        'mmpose.engine.hooks.train_mpjpe_hook',
        'mmpose.engine.optim_wrappers.fixed_amp_optim_wrapper',
        'infra.metrics_csv_hook',
    ],
    allow_failed_imports=False,
)

# ── Architecture constants ───────────────────────────────────────────────────
model_name = 'sapiens_0.3b'
embed_dim = 1024
num_joints = 70
img_h = 640
img_w = 384

# ── Paths ────────────────────────────────────────────────────────────────────
pretrained_checkpoint = (
    '/home/hangyang_umass_edu/MMC/sapiens/pretrain/checkpoints/'
    'sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth')

data_root = __import__('os').environ.get(
    'BEDLAM2_DATA_ROOT',
    '/work/pi_nwycoff_umass_edu/hang/BEDLAM2subset')

# ── Output directory (patched by setup-design) ───────────────────────────────
output_dir = "/work/pi_nwycoff_umass_edu/hang/autosapiens_iter/runs/idea014/design001"

# ── Training schedule ────────────────────────────────────────────────────────
num_epochs = 20 if _stage == 1 else 10
warmup_epochs = 3 if _stage == 1 else 1

train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=5)

# ── Optimizer ────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='FixedAmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.03),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=8,
)

# ── LR Schedule (iteration-based via convert_to_iter_based) ──────────────────
param_scheduler = [
    dict(type='LinearLR', begin=0, end=warmup_epochs, start_factor=0.333,
         by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', begin=warmup_epochs, end=num_epochs,
         eta_min=0, by_epoch=True, convert_to_iter_based=True),
]

# ── Scope ────────────────────────────────────────────────────────────────────
default_scope = 'mmpose'

# ── Visualizer ───────────────────────────────────────────────────────────────
visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

# ── Hooks ────────────────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best=None,
        save_last=True,
        max_keep_ckpts=1,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='TrainMPJPEAveragingHook'),
    dict(type='MetricsCSVHook'),
]

# ── Environment ──────────────────────────────────────────────────────────────
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ── Logger ───────────────────────────────────────────────────────────────────
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = True

# ── Reproducibility ──────────────────────────────────────────────────────────
randomness = dict(seed=2026)

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type='RGBDPose3dEstimator',
    data_preprocessor=dict(type='RGBDPoseDataPreprocessor'),
    backbone=dict(
        type='SapiensBackboneRGBD',
        arch=model_name,
        img_size=(img_h, img_w),
        drop_path_rate=0.1,
        pretrained=pretrained_checkpoint,
    ),
    head=dict(
        type='Pose3dTransformerHead',
        in_channels=embed_dim,
        hidden_dim=256,
        num_joints=num_joints,
        num_heads=8,
        dropout=0.1,
        loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                         loss_weight=1.0),
        loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                        loss_weight=1.0),
        loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                     loss_weight=1.0),
        loss_weight_depth=1.0,
        loss_weight_uv=1.0,
        depth_head_type='classification',
        num_depth_bins=64,
        depth_range_min=1.0,
        depth_range_max=15.0,
        depth_soft_label_sigma=1.5,
        depth_aux_reg_weight=0.0,
    ),
    test_cfg=dict(flip_test=False),
)

# ── Data Pipelines ───────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadBedlamLabels', depth_required=True),
    dict(type='NoisyBBoxTransform'),
    dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
    dict(type='SubtractRootJoint'),
    dict(type='PackBedlamInputs',
         meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                    'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
]

val_pipeline = [
    dict(type='LoadBedlamLabels', depth_required=True, filter_invalid=False),
    dict(type='CropPersonRGBD', out_h=img_h, out_w=img_w),
    dict(type='SubtractRootJoint'),
    dict(type='PackBedlamInputs',
         meta_keys=('img_path', 'depth_npy_path', 'folder_name', 'seq_name',
                    'frame_idx', 'body_idx', 'ori_shape', 'img_shape', 'K')),
]

# ── Dataloaders ──────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='Bedlam2Dataset',
        data_root=data_root,
        seq_paths=_splits['train'],
        frame_stride=1,
        pipeline=train_pipeline,
        max_refetch=10,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Bedlam2Dataset',
        data_root=data_root,
        seq_paths=_splits['val'],
        frame_stride=1,
        pipeline=val_pipeline,
        test_mode=True,
        max_refetch=10,
    ),
)

test_dataloader = val_dataloader

# ── Evaluators ───────────────────────────────────────────────────────────────
val_evaluator = dict(type='BedlamMPJPEMetric')
test_evaluator = dict(type='BedlamMPJPEMetric')

# ── Validation / Test cfg ────────────────────────────────────────────────────
val_cfg = dict()
test_cfg = dict()
