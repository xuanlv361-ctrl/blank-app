"""TF-IDF baseline and opt-in local sentence embeddings with versioned caches."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .dataset_profile import ROOT

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Fixed publisher revision; preparation resolves and records an immutable revision.
MODEL_DIR = ROOT / "data" / "processed" / "semantic-model"
FEATURE_DIR = ROOT / "data" / "processed" / "content-features"
CONTENT_VERSION = "canonical-text-v2"
TFIDF_OPTIONS = {"stop_words": "english", "ngram_range": (1, 2), "max_features": 12000, "strip_accents": "unicode"}


class ContentFeatures:
    def __init__(self, texts, backend="tfidf", *, model_dir=None, cache_dir=None):
        self.backend = backend
        texts = list(map(str, texts))
        if backend == "tfidf":
            self.encoder = TfidfVectorizer(**TFIDF_OPTIONS)
            self.matrix = self.encoder.fit_transform(texts)
            self.version = (
                "tfidf:" + hashlib.sha256(json.dumps([CONTENT_VERSION, TFIDF_OPTIONS, texts]).encode()).hexdigest()
            )
            return
        if backend != "minilm":
            raise ValueError("Content backend must be tfidf or minilm")
        model_dir = Path(model_dir or MODEL_DIR)
        manifest = model_dir / "filmtrace-model.json"
        if not manifest.is_file():
            raise ValueError("Semantic model is not prepared. Run python -m src.content_features --prepare first.")
        info = json.loads(manifest.read_text(encoding="utf-8"))
        self.version = f"minilm-v1:{info['revision']}"
        self.encoder = None
        self.model_dir = model_dir
        key = hashlib.sha256(json.dumps([self.version, texts], ensure_ascii=True).encode()).hexdigest()
        root = Path(cache_dir or FEATURE_DIR)
        target = root / f"{key}.npz"
        if target.is_file():
            with np.load(target, allow_pickle=False) as data:
                matrix = data["features"]
            if matrix.shape != (len(texts), 384) or not np.isfinite(matrix).all():
                raise ValueError("Invalid embedding cache; remove this cache artifact and prepare again")
            self.matrix = matrix
        else:
            self.matrix = self.transform(texts)
            root.mkdir(parents=True, exist_ok=True)
            temporary = root / f"{key}.{uuid.uuid4().hex}.tmp.npz"
            try:
                np.savez_compressed(temporary, features=self.matrix)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    def transform(self, texts):
        if self.backend == "tfidf":
            return self.encoder.transform(texts)
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install requirements-semantic.txt to enable MiniLM") from exc
            self.encoder = SentenceTransformer(
                str(self.model_dir), device="cpu", local_files_only=True, trust_remote_code=False
            )
        return np.asarray(
            self.encoder.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def prepare_model():
    """Explicit outbound preparation only; normal serving never downloads models."""
    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    revision = HfApi().model_info(MODEL_NAME).sha
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        revision=revision,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
    )
    info = {"model": MODEL_NAME, "revision": revision, "license": "Apache-2.0", "dimensions": 384}
    (MODEL_DIR / "filmtrace-model.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.parse_args()
    prepare_model()
