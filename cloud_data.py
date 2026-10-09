"""Read an audited public-only data bundle without network requests or extraction."""

import base64
import gzip
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_movies, clean_ratings

ROOT = Path(__file__).resolve().parent


def load_public_data():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    packed = base64.b64decode((ROOT / "data/public-data.gz.b64").read_bytes(), validate=True)
    if hashlib.sha256(packed).hexdigest() != manifest["sha256"]:
        raise ValueError("Public dataset checksum mismatch")
    payload = json.loads(gzip.decompress(packed))
    ratings = clean_ratings(pd.read_csv(io.StringIO(payload["ratings_csv"])))
    movies = clean_movies(pd.read_csv(io.StringIO(payload["movies_csv"])))
    assert len(ratings) == manifest["ratings"] and len(movies) == manifest["movies"]
    return ratings, movies, payload["discovery"], manifest


def poster_url(value):
    if isinstance(value, str) and re.fullmatch(r"https://image\.tmdb\.org/t/p/w500/[A-Za-z0-9_.-]+", value):
        return value.replace("/w500/", "/w342/")
    return None
