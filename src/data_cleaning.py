"""Data cleaning and validation utilities for MovieLens 100K."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .recsys_service import NON_GENRE_COLUMNS as METADATA_COLUMNS

NON_GENRE_COLUMNS = {
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "video_release_date",
    "imdb_url",
}
NON_GENRE_COLUMNS |= METADATA_COLUMNS


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Return a validated ratings table suitable for modeling."""
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing required rating columns: {sorted(missing)}")

    cleaned = ratings.copy()
    for col in ["user_id", "movie_id", "rating", "timestamp"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    undated = ratings.attrs.get("timestamps_available") is False
    required_values = ["user_id", "movie_id", "rating"]
    if not undated:
        required_values.append("timestamp")
    cleaned = cleaned.dropna(subset=required_values)
    cleaned = cleaned[cleaned["rating"].between(0.5, 5)]
    if not undated:
        cleaned = cleaned[cleaned["timestamp"] > 0]

    int_cols = ["user_id", "movie_id"]
    cleaned[int_cols] = cleaned[int_cols].astype(int)
    cleaned["timestamp"] = cleaned["timestamp"].astype("Int64")

    cleaned = (
        cleaned.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "movie_id"], keep="last")
        .reset_index(drop=True)
    )
    cleaned.attrs["timestamps_available"] = not undated
    return cleaned


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned movie metadata with valid IDs and binary genre columns."""
    required = {"movie_id", "title"}
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"Missing required movie columns: {sorted(missing)}")

    cleaned = movies.copy()
    cleaned["movie_id"] = pd.to_numeric(cleaned["movie_id"], errors="coerce")
    cleaned = cleaned.dropna(subset=["movie_id", "title"])
    cleaned["movie_id"] = cleaned["movie_id"].astype(int)
    cleaned["title"] = cleaned["title"].astype(str).str.strip()
    cleaned = cleaned[cleaned["title"] != ""]

    genre_cols = [col for col in cleaned.columns if col not in NON_GENRE_COLUMNS]
    for col in genre_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0).astype(int)
        cleaned[col] = cleaned[col].clip(lower=0, upper=1)

    if "release_date" in cleaned.columns:
        cleaned["release_year"] = cleaned.apply(_extract_year, axis=1)

    cleaned = cleaned.drop_duplicates(subset=["movie_id"], keep="first").reset_index(drop=True)
    return cleaned


def align_ratings_with_movies(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only ratings whose movie IDs exist in the movie metadata table."""
    movie_ids = set(movies["movie_id"])
    aligned_ratings = ratings[ratings["movie_id"].isin(movie_ids)].reset_index(drop=True)
    rated_movie_ids = set(aligned_ratings["movie_id"])
    aligned_movies = movies[movies["movie_id"].isin(rated_movie_ids)].reset_index(drop=True)
    return aligned_ratings, aligned_movies


def data_quality_report(
    raw_ratings: pd.DataFrame,
    clean_ratings_df: pd.DataFrame,
    raw_movies: pd.DataFrame,
    clean_movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize cleaning effects for the notebook/report."""
    rows = [
        {
            "table": "ratings",
            "raw_rows": len(raw_ratings),
            "clean_rows": len(clean_ratings_df),
            "removed_rows": len(raw_ratings) - len(clean_ratings_df),
            "missing_values_raw": int(raw_ratings.isna().sum().sum()),
            "duplicate_keys_raw": int(raw_ratings.duplicated(["user_id", "movie_id"]).sum()),
        },
        {
            "table": "movies",
            "raw_rows": len(raw_movies),
            "clean_rows": len(clean_movies_df),
            "removed_rows": len(raw_movies) - len(clean_movies_df),
            "missing_values_raw": int(raw_movies.isna().sum().sum()),
            "duplicate_keys_raw": int(raw_movies.duplicated(["movie_id"]).sum()),
        },
    ]
    return pd.DataFrame(rows)


def _extract_year(row: pd.Series) -> float:
    release_date = row.get("release_date")
    if isinstance(release_date, str):
        match = re.search(r"(\d{4})", release_date)
        if match:
            return float(match.group(1))

    title = row.get("title")
    if isinstance(title, str):
        match = re.search(r"\((\d{4})\)", title)
        if match:
            return float(match.group(1))
    return np.nan
