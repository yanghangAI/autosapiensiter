"""Custom MMEngine hook that writes epoch and iteration metrics to CSV files."""

from __future__ import annotations

import csv
import os
from typing import Optional, Sequence

from mmpose.registry import HOOKS
from mmengine.hooks import Hook

# Metric name mapping: MMEngine evaluator key -> CSV column name
_METRIC_MAP = {
    'composite/val': 'composite_val',
    'mpjpe/body/val': 'mpjpe_body_val',
    'mpjpe/pelvis/val': 'mpjpe_pelvis_val',
    'mpjpe/rel/val': 'mpjpe_rel_val',
    'mpjpe/hand/val': 'mpjpe_hand_val',
    'mpjpe/abs/val': 'mpjpe_abs_val',
}

_EPOCH_COLS = [
    'epoch', 'composite_val', 'mpjpe_body_val', 'mpjpe_pelvis_val',
    'mpjpe_rel_val', 'mpjpe_hand_val', 'mpjpe_abs_val',
]

_ITER_COLS = [
    'iter', 'epoch', 'loss_joints_train', 'loss_depth_train', 'loss_uv_train',
]

# Loss key mapping: runner log key -> CSV column
_LOSS_MAP = {
    'loss/joints/train': 'loss_joints_train',
    'loss/depth/train': 'loss_depth_train',
    'loss/uv/train': 'loss_uv_train',
}


def _ensure_header(path: str, columns: list[str]) -> None:
    """Write CSV header if the file does not exist or is empty."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)


def _append_row(path: str, row: list) -> None:
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


@HOOKS.register_module()
class MetricsCSVHook(Hook):
    """Write training and validation metrics to CSV files.

    - ``metrics.csv``: one row per validation epoch
    - ``iter_metrics.csv``: one row per training iteration
    """

    priority = 'VERY_LOW'  # run after other hooks have updated metrics

    def __init__(self) -> None:
        super().__init__()
        self._epoch_csv_ready = False
        self._iter_csv_ready = False

    def _epoch_csv_path(self, runner) -> str:
        return os.path.join(runner.work_dir, 'metrics.csv')

    def _iter_csv_path(self, runner) -> str:
        return os.path.join(runner.work_dir, 'iter_metrics.csv')

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        """Write a row to metrics.csv after each validation epoch."""
        if metrics is None:
            return

        path = self._epoch_csv_path(runner)
        if not self._epoch_csv_ready:
            _ensure_header(path, _EPOCH_COLS)
            self._epoch_csv_ready = True

        # runner.epoch is already 1-indexed here because after_train_epoch
        # increments it before validation runs
        epoch = runner.epoch

        # Map metric names
        csv_vals = {}
        for engine_key, csv_key in _METRIC_MAP.items():
            csv_vals[csv_key] = metrics.get(engine_key, '')

        # Fallback computation for composite_val
        if csv_vals.get('composite_val', '') == '':
            body = csv_vals.get('mpjpe_body_val', '')
            pelvis = csv_vals.get('mpjpe_pelvis_val', '')
            if body != '' and pelvis != '':
                csv_vals['composite_val'] = 0.67 * float(body) + 0.33 * float(pelvis)

        row = [epoch]
        for col in _EPOCH_COLS[1:]:
            val = csv_vals.get(col, '')
            if isinstance(val, float):
                row.append(f'{val:.4f}')
            else:
                row.append(val)

        _append_row(path, row)

    def after_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
        outputs=None,
    ) -> None:
        """Write a row to iter_metrics.csv after each training iteration."""
        path = self._iter_csv_path(runner)
        if not self._iter_csv_ready:
            _ensure_header(path, _ITER_COLS)
            self._iter_csv_ready = True

        current_iter = runner.iter + 1  # 1-indexed
        epoch = runner.epoch + 1

        # Extract losses from runner's log buffer
        log_buffer = runner.message_hub.log_scalars
        row_vals = {'iter': current_iter, 'epoch': epoch}
        for engine_key, csv_key in _LOSS_MAP.items():
            tag = f'train/{engine_key}'
            if tag in log_buffer:
                val = log_buffer[tag].current()
                row_vals[csv_key] = f'{val:.6f}'
            elif engine_key in log_buffer:
                val = log_buffer[engine_key].current()
                row_vals[csv_key] = f'{val:.6f}'
            else:
                row_vals[csv_key] = ''

        row = [row_vals.get(col, '') for col in _ITER_COLS]
        _append_row(path, row)
