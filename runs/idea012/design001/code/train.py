#!/usr/bin/env python3
"""Training wrapper for the automation framework.

Stage is selected via the STAGE env var (1 or 2). Output is written to
``<design>/output/stage{N}/``. AMP is enabled via the config's
FixedAmpOptimWrapper; we do NOT pass --no-amp.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, 'config.py')

if os.path.basename(HERE) == 'code':
    DESIGN_DIR = os.path.dirname(HERE)
else:
    DESIGN_DIR = HERE

STAGE = os.environ.get('STAGE', '1')
WORK_DIR = os.path.join(DESIGN_DIR, 'output', f'stage{STAGE}')
os.makedirs(WORK_DIR, exist_ok=True)

TOOLS_TRAIN = '/home/hangyang_umass_edu/MMC/sapiens/pose/tools/train.py'

sys.exit(subprocess.call([
    sys.executable, TOOLS_TRAIN, CONFIG,
    '--work-dir', WORK_DIR,
]))
