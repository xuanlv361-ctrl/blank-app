"""Simple baseline recommenders for rating prediction and top-k ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cancellation import checkpoint
from .config import RANDOM_STATE, RATING_SCALE
from .metrics import clip_predictions
from .preprocessing import global_mean


class GlobalMeanBaseline:
    """Predict every rating with the training-set global mean."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0

    def fit(self, train: pd.DataFrame) -> GlobalMeanBaseline:
        """Fit the model on the training ratings and return self."""
        self.global_mean_ = global_mean(train)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        return self.global_mean_

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        return np.full(len(test), self.global_mean_, dtype=float)


class UserMeanBaseline:
    """Predict with each user's mean rating, falling back to the global mean."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.user_means_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> UserMeanBaseline:
        """Fit the model on the training ratings and return self."""
        self.global_mean_ = global_mean(train)
        self.user_means_ = train.groupby("user_id")["rating"].mean()
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        if self.user_means_ is None:
            raise RuntimeError("Call fit() before predict().")
        return float(self.user_means_.get(user_id, self.global_mean_))

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        return np.array(
            [self.predict(int(row.user_id), int(row.movie_id)) for row in test.itertuples(index=False)],
            dtype=float,
        )


class ItemMeanBaseline:
    """Predict with each movie's mean rating, falling back to the global mean."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.item_means_: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> ItemMeanBaseline:
        """Fit the model on the training ratings and return self."""
        self.global_mean_ = global_mean(train)
        self.item_means_ = train.groupby("movie_id")["rating"].mean()
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        if self.item_means_ is None:
            raise RuntimeError("Call fit() before predict().")
        return float(self.item_means_.get(movie_id, self.global_mean_))

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        return np.array(
            [self.predict(int(row.user_id), int(row.movie_id)) for row in test.itertuples(index=False)],
            dtype=float,
        )


class BiasBaseline:
    """
    Regularized user-item bias baseline.

    Prediction formula:
        global_mean + user_bias[user_id] + item_bias[movie_id]

    Bias terms are estimated by alternating updates on the training set only.
    """

    def __init__(
        self,
        n_epochs: int = 20,
        reg: float = 10.0,
    ) -> None:
        self.n_epochs = n_epochs
        self.reg = reg
        self.global_mean_: float = 0.0
        self.user_bias_: dict[int, float] = {}
        self.item_bias_: dict[int, float] = {}

    def fit(self, train: pd.DataFrame, *, cancel=None) -> BiasBaseline:
        """Fit the model on the training ratings and return self."""
        self.global_mean_ = global_mean(train)
        user_groups = {
            int(user_id): group[["movie_id", "rating"]].copy() for user_id, group in train.groupby("user_id")
        }
        item_groups = {
            int(movie_id): group[["user_id", "rating"]].copy() for movie_id, group in train.groupby("movie_id")
        }

        self.user_bias_ = {user_id: 0.0 for user_id in user_groups}
        self.item_bias_ = {movie_id: 0.0 for movie_id in item_groups}

        for _ in range(self.n_epochs):
            checkpoint(cancel)
            for user_id, group in user_groups.items():
                checkpoint(cancel)
                residual = 0.0
                for row in group.itertuples(index=False):
                    residual += float(row.rating) - self.global_mean_ - self.item_bias_.get(int(row.movie_id), 0.0)
                self.user_bias_[user_id] = residual / (self.reg + len(group))

            for movie_id, group in item_groups.items():
                checkpoint(cancel)
                residual = 0.0
                for row in group.itertuples(index=False):
                    residual += float(row.rating) - self.global_mean_ - self.user_bias_.get(int(row.user_id), 0.0)
                self.item_bias_[movie_id] = residual / (self.reg + len(group))
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        pred = self.global_mean_ + self.user_bias_.get(user_id, 0.0) + self.item_bias_.get(movie_id, 0.0)
        return float(clip_predictions(np.array([pred]), *RATING_SCALE)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        return np.array(
            [self.predict(int(row.user_id), int(row.movie_id)) for row in test.itertuples(index=False)],
            dtype=float,
        )


class RandomRatingBaseline:
    """Random rating predictor sampled uniformly from the valid rating scale."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state
        self.rng_: np.random.Generator | None = None

    def fit(self, train: pd.DataFrame) -> RandomRatingBaseline:
        """Fit the model on the training ratings and return self."""
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        if self.rng_ is None:
            raise RuntimeError("Call fit() before predict().")
        low, high = RATING_SCALE
        return float(self.rng_.integers(low, high + 1))

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        if self.rng_ is None:
            raise RuntimeError("Call fit() before predict_batch().")
        low, high = RATING_SCALE
        preds = self.rng_.integers(low, high + 1, size=len(test)).astype(float)
        return clip_predictions(preds, *RATING_SCALE)


class MostPopularBaseline:
    """Recommend globally popular movies, excluding items already seen by a user."""

    def __init__(self, min_ratings: int = 1) -> None:
        self.min_ratings = min_ratings
        self.ranked_items_: list[int] = []
        self.user_history_: dict[int, set[int]] = {}

    def fit(self, train: pd.DataFrame) -> MostPopularBaseline:
        """Fit the model on the training ratings and return self."""
        popularity = (
            train.groupby("movie_id")
            .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
            .sort_values(["rating_count", "mean_rating"], ascending=[False, False])
        )
        popularity = popularity[popularity["rating_count"] >= self.min_ratings]
        self.ranked_items_ = [int(movie_id) for movie_id in popularity.index]
        self.user_history_ = train.groupby("user_id")["movie_id"].apply(lambda values: set(map(int, values))).to_dict()
        return self

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Return the top-N recommended movie ids for the given user."""
        seen = self.user_history_.get(user_id, set())
        recs = [movie_id for movie_id in self.ranked_items_ if movie_id not in seen]
        return recs[:k]

    def recommend_batch(self, user_ids: list[int], k: int = 10) -> dict[int, list[int]]:
        """Return recommended movie ids for each user in the batch."""
        return {int(user_id): self.recommend(int(user_id), k=k) for user_id in user_ids}
