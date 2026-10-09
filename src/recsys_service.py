"""Shared recommendation-service helpers used by both the Streamlit app
(`app.py`) and the PyQt5 desktop app (`qt_app.py`).

This module consolidates logic that was previously duplicated between the
two front ends: model construction, calibrated score blending, genre
helpers, and popularity ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RATING_SCALE
from .item_based_cf import ItemBasedCF
from .svd_model import SVDRecommender
from .user_based_cf import UserBasedCF

NON_GENRE_COLUMNS = {
    "catalog_source_key",
    "movie_id",
    "title",
    "release_date",
    "release_year",
    "video_release_date",
    "imdb_url",
    "poster_filename",
    "source_movie_id", "source", "source_title", "year_source", "metadata_status",
    "tmdb_id", "overview", "poster_url", "tmdb_vote_average", "tmdb_vote_count",
}

# Algorithm display names used across both front ends, normalized to a
# common set of keys for model construction.
_USER_CF_NAMES = {"user-based cf", "user-based collaborative filtering"}
_SVD_NAMES = {"svd", "svd matrix factorization", "svd (truncatedsvd)"}


def build_model(
    ratings: pd.DataFrame,
    algorithm: str,
    k: int = 20,
    metric: str = "cosine",
    factors: int = 50,
    *,
    cancel=None,
) -> object:
    """Fit and return a recommendation model for the given algorithm name.

    Recognizes both the short labels used by the Qt app ("User-based CF",
    "SVD", "Item-based CF") and the longer labels used by the Streamlit app
    ("User-based Collaborative Filtering", "SVD Matrix Factorization").
    Anything else falls back to item-based CF.
    """
    name = algorithm.strip().lower()
    if name in _USER_CF_NAMES:
        return UserBasedCF(k=k, metric=metric).fit(ratings, cancel=cancel)
    if name in _SVD_NAMES:
        return SVDRecommender(n_components=factors).fit(ratings, cancel=cancel)
    return ItemBasedCF(k=k, metric=metric).fit(ratings, cancel=cancel)


def calibrated_score(
    raw_pred: float,
    avg_rating: float,
    user_mean: float,
    confidence: float,
    rating_count: int,
    global_mean: float,
) -> float:
    """Shrink a raw model prediction toward reliable evidence for display/ranking.

    Blends the raw prediction with the movie's average rating and the
    user's average rating, weighted by how confident/well-supported the
    prediction is, then shrinks toward the global mean for low-evidence
    cases. The result is clipped to a display-friendly range.
    """
    count_weight = min(1.0, np.log1p(max(rating_count, 0)) / np.log1p(250))
    evidence = 0.58 * raw_pred + 0.27 * avg_rating + 0.15 * user_mean
    reliability = 0.72 + 0.20 * confidence + 0.08 * count_weight
    score = reliability * evidence + (1 - reliability) * global_mean
    return float(np.clip(score, RATING_SCALE[0], 4.99))


def calibrated_score_batch(
    raw_pred: np.ndarray,
    avg_rating: np.ndarray,
    user_mean: float,
    confidence: np.ndarray,
    rating_count: np.ndarray,
    global_mean: float,
) -> np.ndarray:
    """Vectorized variant of `calibrated_score` for scoring many candidates at once."""
    count_weight = np.minimum(1.0, np.log1p(np.maximum(rating_count, 0)) / np.log1p(250))
    evidence = 0.58 * raw_pred + 0.27 * avg_rating + 0.15 * user_mean
    reliability = 0.72 + 0.20 * confidence + 0.08 * count_weight
    score = reliability * evidence + (1 - reliability) * global_mean
    return np.clip(score, RATING_SCALE[0], 4.99)


def recommendation_reason(algorithm: str, genres: str) -> str:
    """Return a natural-language recommendation reason for the given algorithm and genres."""
    name = algorithm.strip().lower()
    primary_genre = genres.split(",")[0].strip() if genres else "similar"
    if name in _USER_CF_NAMES:
        return "Recommended because users with similar rating behavior gave this movie strong scores."
    if name in _SVD_NAMES:
        return "Recommended because latent-factor patterns indicate a high personalized preference score."
    if not genres or genres.strip().casefold() in {"unknown", "unclassified", "未分类", "未知"}:
        return "Recommended from similarity between movie rating patterns."
    return f"Recommended because your rated movies include similar {primary_genre} titles."


def genre_columns(movies: pd.DataFrame) -> list[str]:
    """Return the list of genre indicator columns present in the movies frame."""
    return [col for col in movies.columns if col not in NON_GENRE_COLUMNS]


def genre_text(movie_row: pd.Series, genre_cols: list[str]) -> str:
    """Return a comma-separated genre string for one movie row."""
    genres = [genre for genre in genre_cols if int(movie_row.get(genre, 0) or 0) == 1]
    return ", ".join(genres) if genres else "Unknown"


def popular_movies(
    catalog: pd.DataFrame,
    rank_by: str,
    top_n: int,
    min_ratings_for_avg: int = 0,
) -> pd.DataFrame:
    """Rank a movie catalog by rating count, average rating, or a popularity-shrunk
    weighted score (Bayesian average using the catalog's 75th-percentile rating count
    as the prior weight).

    `min_ratings_for_avg` optionally filters out low-evidence movies before
    ranking by raw "Average Rating" (the Qt admin stats view uses 50).
    """
    ranked = catalog.copy()
    global_mean = ranked["Average Rating"].mean()
    min_votes = max(1, int(ranked["Ratings"].quantile(0.75)))
    ranked["Weighted Score"] = (
        ranked["Ratings"] / (ranked["Ratings"] + min_votes) * ranked["Average Rating"]
        + min_votes / (ranked["Ratings"] + min_votes) * global_mean
    ).round(3)
    if rank_by == "Number of Ratings":
        sort_cols = ["Ratings", "Average Rating"]
    elif rank_by == "Average Rating":
        if min_ratings_for_avg > 0:
            ranked = ranked[ranked["Ratings"] >= min_ratings_for_avg]
        sort_cols = ["Average Rating", "Ratings"]
    else:
        sort_cols = ["Weighted Score", "Ratings"]
    return ranked.sort_values(sort_cols, ascending=False).head(top_n)
