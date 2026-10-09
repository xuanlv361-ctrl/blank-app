"""Startup-only dataset selection, shared by the web and desktop applications."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATASETS = {"movielens-100k": "MovieLens 100K", "letterboxd-2025": "Letterboxd 2025",
            "movielens-32m-21c": "MovieLens 32M (21st century)"}
DATASET = os.environ.get("FILMTRACE_DATASET", "movielens-100k").strip()
if DATASET not in DATASETS:
    raise ValueError(f"FILMTRACE_DATASET must be one of: {', '.join(DATASETS)}")

DATASET_LABEL = DATASETS[DATASET]
MODERN_DIR = ROOT / "data" / "processed" / "letterboxd-2025"
IS_SNAPSHOT = DATASET != "movielens-100k"
SNAPSHOT_DIR = ROOT / "data" / "processed" / DATASET
HAS_TIMESTAMPS = DATASET != "letterboxd-2025"
SNAPSHOT_INFO = {}
if IS_SNAPSHOT and (SNAPSHOT_DIR / "active.json").exists():
    SNAPSHOT_INFO = json.loads((SNAPSHOT_DIR / "active.json").read_text(encoding="utf-8"))
