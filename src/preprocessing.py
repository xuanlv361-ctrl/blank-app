"""Train/test split and rating-matrix utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


def evaluation_split(ratings: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Use chronology only when genuine timestamps are available."""
    if "timestamp" in ratings and ratings["timestamp"].notna().all():
        return user_temporal_split(ratings, test_size=test_size)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    train, test = [], []
    for _, group in ratings.sort_values(["user_id", "movie_id"]).groupby("user_id"):
        order = rng.permutation(len(group))
        count = min(max(1, round(len(group) * test_size)), len(group) - 1)
        test.append(group.iloc[order[:count]])
        train.append(group.iloc[order[count:]])
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def rating_date(value, format_string: str = "%Y-%m-%d") -> str:
    """Never turn missing source timestamps into 1970 or the import date."""
    if pd.isna(value):
        return "Not provided"
    return pd.to_datetime(value, unit="s").strftime(format_string)


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random hold-out split on rating rows."""
    train, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def time_based_split(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological hold-out split to reduce future-to-past leakage."""
    if timestamp_col not in ratings.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if ratings[timestamp_col].isna().any():
        raise ValueError("Chronological evaluation requires real timestamps; use evaluation_split.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    ordered = ratings.sort_values(timestamp_col).reset_index(drop=True)
    split_at = int(len(ordered) * (1 - test_size))
    train = ordered.iloc[:split_at].copy()
    test = ordered.iloc[split_at:].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def user_temporal_split(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    timestamp_col: str = "timestamp",
    user_col: str = "user_id",
    min_train_ratings: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each user's ratings chronologically into train and test rows.

    This preserves each user's past-to-future ordering while ensuring every test
    user has training history, which is important for evaluating collaborative
    filtering without turning the main experiment into a pure cold-start task.
    """
    if timestamp_col not in ratings.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if ratings[timestamp_col].isna().any():
        raise ValueError("Chronological evaluation requires real timestamps; use evaluation_split.")
    if user_col not in ratings.columns:
        raise ValueError(f"Missing user column: {user_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if min_train_ratings < 1:
        raise ValueError("min_train_ratings must be at least 1.")

    train_parts = []
    test_parts = []
    ordered = ratings.sort_values([user_col, timestamp_col, "movie_id"]).reset_index(drop=True)

    for _, user_rows in ordered.groupby(user_col, sort=False):
        n_rows = len(user_rows)
        n_test = max(1, int(round(n_rows * test_size))) if n_rows > min_train_ratings else 0
        n_test = min(n_test, max(0, n_rows - min_train_ratings))

        if n_test == 0:
            train_parts.append(user_rows)
            continue

        train_parts.append(user_rows.iloc[:-n_test])
        test_parts.append(user_rows.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def user_temporal_train_val_test_split(
    ratings: pd.DataFrame,
    val_size: float = TEST_SIZE,
    test_size: float = TEST_SIZE,
    timestamp_col: str = "timestamp",
    user_col: str = "user_id",
    min_train_ratings: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each user's ratings chronologically into train, validation, and test rows.

    For each user (sorted by timestamp), the most recent ratings are carved off
    for the test set, then the next most recent for the validation set, leaving
    the remainder for training. Hyperparameter tuning should use only the
    validation split; the test split must remain untouched until the final
    evaluation.
    """
    if timestamp_col not in ratings.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if ratings[timestamp_col].isna().any():
        raise ValueError("Chronological evaluation requires real timestamps.")
    if user_col not in ratings.columns:
        raise ValueError(f"Missing user column: {user_col}")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")
    if min_train_ratings < 1:
        raise ValueError("min_train_ratings must be at least 1.")

    train_parts = []
    val_parts = []
    test_parts = []
    ordered = ratings.sort_values([user_col, timestamp_col]).reset_index(drop=True)

    for _, user_rows in ordered.groupby(user_col, sort=False):
        n_rows = len(user_rows)
        n_test = max(1, int(round(n_rows * test_size))) if n_rows > min_train_ratings else 0
        n_test = min(n_test, max(0, n_rows - min_train_ratings))

        remaining = n_rows - n_test
        n_val = max(1, int(round(n_rows * val_size))) if remaining > min_train_ratings else 0
        n_val = min(n_val, max(0, remaining - min_train_ratings))

        n_train = n_rows - n_test - n_val

        train_parts.append(user_rows.iloc[:n_train])
        if n_val > 0:
            val_parts.append(user_rows.iloc[n_train : n_train + n_val])
        if n_test > 0:
            test_parts.append(user_rows.iloc[n_train + n_val :])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else ratings.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def global_mean(train: pd.DataFrame) -> float:
    """Return the global mean rating over the given ratings."""
    return float(train["rating"].mean())


def user_mean_ratings(train: pd.DataFrame) -> pd.Series:
    """Return a Series mapping each user_id to that user's mean rating."""
    return train.groupby("user_id")["rating"].mean()


def item_mean_ratings(train: pd.DataFrame) -> pd.Series:
    """Return a Series mapping each movie_id to that movie's mean rating."""
    return train.groupby("movie_id")["rating"].mean()


def build_user_item_matrix(
    train: pd.DataFrame,
    fill_value: float = np.nan,
) -> pd.DataFrame:
    """Pivot to users (rows) × movies (columns)."""
    return train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        fill_value=fill_value,
    )


def mean_center_by_user(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Subtract per-user mean (ignoring NaN)."""
    user_means = matrix.mean(axis=1)
    centered = matrix.sub(user_means, axis=0)
    return centered, user_means


def mean_center_by_item(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Subtract per-item mean (ignoring NaN)."""
    item_means = matrix.mean(axis=0)
    centered = matrix.sub(item_means, axis=1)
    return centered, item_means
