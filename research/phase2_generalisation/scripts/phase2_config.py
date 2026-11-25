"""Runtime paths for Phase 2. Dataset images must be obtained from their original sources."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / 'research' / 'phase2_generalisation'
RESULTS = PACKAGE / 'results'
OUT = Path(os.environ.get('PHASE2_OUTPUT_DIR', str(PACKAGE / 'reproduced_outputs'))).expanduser()
JIC_DIR = Path(os.environ.get('PHASE2_JIC_DIR', str(REPO / 'external_data' / 'JIC' / 'val'))).expanduser()
NUST_DIR = Path(os.environ.get('PHASE2_NUST_DIR', str(REPO / 'external_data' / 'NUST' / 'NWRD'))).expanduser()
MODEL = Path(os.environ.get('PHASE2_MODEL', str(REPO / 'models' / 'wheat_rust_mobilenetv2.keras'))).expanduser()
JIC_REFERENCE = Path(os.environ.get('PHASE2_JIC_REFERENCE', str(RESULTS / 'jic_predictions.csv'))).expanduser()
# The unmodified original CSV preserves the SHA256 referenced in Experiment 7.
NUST_REFERENCE = Path(os.environ.get('PHASE2_NUST_REFERENCE', str(RESULTS / 'phase2_nust_evaluation_nust_frozen_predictions.csv'))).expanduser()

def nust_path(value):
    """Resolve both original Windows and portable NUST image/mask CSV paths."""
    s = str(value).replace('\\', '/')
    if '/NWRD/' in s:
        return NUST_DIR / s.split('/NWRD/', 1)[1]
    if s.startswith('NUST/NWRD/'):
        return NUST_DIR / s[len('NUST/NWRD/'):]
    if s.startswith(('train/images/', 'train/masks/', 'test/images/', 'test/masks/')):
        return NUST_DIR / s
    raise ValueError(f'Unrecognized NUST path (must be within NWRD): {value}')
