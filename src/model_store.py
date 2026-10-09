"""Versioned local model cache shared by web, Qt and offline evaluation."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import joblib
import pandas as pd

from .cancellation import Cancelled, checkpoint
from .cf_capacity import admit_cf_fit
from .dataset_profile import ROOT
from .recsys_service import _SVD_NAMES, _USER_CF_NAMES, build_model

MODEL_VERSION = "personal-v4-stable-pearson"
_models = OrderedDict()
_lock = threading.RLock()
_inflight = {}
_training_gate = threading.Semaphore(1)
__all__ = ["Cancelled", "checkpoint", "get_model", "fingerprint", "cache_key", "MODEL_VERSION"]


def fingerprint(frame: pd.DataFrame) -> str:
    cols = sorted(frame.columns)
    ordered = frame[cols].sort_values(cols, na_position="last").reset_index(drop=True)
    return hashlib.sha256(ordered.to_json(orient="split", date_format="iso", double_precision=15).encode()).hexdigest()


def cache_key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def get_model(
    ratings,
    algorithm="Item-based CF",
    k=20,
    metric="cosine",
    factors=50,
    *,
    generation=0,
    progress=None,
    cancel=None,
    cache_dir=None,
):
    checkpoint(cancel)
    if k < 1 or factors < 1 or metric not in {"cosine", "pearson"}:
        raise ValueError("Invalid model parameters")
    key = cache_key(MODEL_VERSION, fingerprint(ratings), algorithm, k, metric, factors, generation)
    root = Path(cache_dir) if cache_dir is not None else ROOT / "data" / "processed" / "model-cache"
    identity = (str(root.resolve()), key)
    name = algorithm.strip().lower()
    if name not in _SVD_NAMES:
        admit_cf_fit(ratings, "user" if name in _USER_CF_NAMES else "item", stage="CF before cache model loading")
    # The metadata lock never covers disk IO or training. Only identical keys wait.
    while True:
        checkpoint(cancel)
        with _lock:
            if identity in _models:
                _models.move_to_end(identity)
                return _models[identity], {"cache": "memory", "seconds": 0.0, "key": key}
            pending = _inflight.get(identity)
            if pending is None:
                pending = threading.Event()
                _inflight[identity] = pending
                break
        pending.wait(0.05)
    try:
        root.mkdir(parents=True, exist_ok=True)
        target = root / f"{key}.joblib"
        start = time.perf_counter()
        if progress:
            progress(f"Loading / training {algorithm}")
        if target.is_file():
            # Only artifacts created locally by this application are read here.
            try:
                model = joblib.load(target)
                source = "disk"
            except (OSError, ValueError, EOFError, KeyError):
                model = None
        else:
            model = None
        if model is None:
            # Bound concurrent cold fits without blocking unrelated warm/disk hits.
            while not _training_gate.acquire(timeout=0.05):
                checkpoint(cancel)
            try:
                checkpoint(cancel)
                model = build_model(ratings, algorithm, k, metric, factors, cancel=cancel)
            finally:
                _training_gate.release()
            checkpoint(cancel)
            temporary = root / f"{key}.{uuid.uuid4().hex}.tmp"
            try:
                joblib.dump(model, temporary)
                checkpoint(cancel)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            source = "trained"
        checkpoint(cancel)
        with _lock:
            checkpoint(cancel)
            _models[identity] = model
            while len(_models) > 6:
                _models.popitem(last=False)
        return model, {"cache": source, "seconds": time.perf_counter() - start, "key": key}
    finally:
        with _lock:
            _inflight.pop(identity, None)
            pending.set()
