"""Central place for paths and global constants.

Why: every script/notebook imports paths from here, so nothing depends on
which folder you launched Python from (no more "../data/raw").
"""
from pathlib import Path

# this file lives in <project>/src/config.py  ->  parents[1] is <project>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"          # read-only, never write here
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
CWRU_RAW = DATA_RAW / "cwru"

SEED = 42
TARGET_FS = 12_000   # every CWRU signal is brought to 12 kHz before any analysis
