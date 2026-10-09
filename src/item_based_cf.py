"""Item-based collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cancellation import checkpoint
from .cf_capacity import admit_cf_fit, model_array_bytes
from .config import K_NEIGHBORS, MIN_COMMON_RATINGS, RATING_SCALE, SIMILARITY_METRIC
from .metrics import clip_predictions
from .preprocessing import build_user_item_matrix, global_mean, mean_center_by_item
from .similarity import pairwise_similarity_matrix


class ItemBasedCF:
    """Predict ratings from similar items' centered ratings."""

    def __init__(
        self,
        k: int = K_NEIGHBORS,
        min_common: int = MIN_COMMON_RATINGS,
        metric: str = SIMILARITY_METRIC,
        *,
        memory_budget_mb: float | None = None,
        block_size: int | None = None,
    ):
        self.k = k
        self.min_common = min_common
        self.metric = metric
        self.memory_budget_mb = memory_budget_mb
        self.block_size = block_size
        self._matrix: pd.DataFrame | None = None
        self._centered: pd.DataFrame | None = None
        self._item_means: pd.Series | None = None
        self._sim: pd.DataFrame | None = None
        self._global_mean: float = 0.0

    def fit(self, train: pd.DataFrame, *, cancel=None) -> ItemBasedCF:
        """Fit the model on the training ratings and return self."""
        checkpoint(cancel)
        budget = getattr(self, "memory_budget_mb", None)
        block_size = getattr(self, "block_size", None)
        retained = model_array_bytes(self)
        admit_cf_fit(train, "item", memory_budget_mb=budget, block_size=block_size, resident_bytes=retained)
        overall_mean = global_mean(train)
        matrix = build_user_item_matrix(train, fill_value=np.nan)
        centered, item_means = mean_center_by_item(matrix)
        similarities = pairwise_similarity_matrix(
            centered.T,
            metric=self.metric,
            min_common=self.min_common,
            memory_budget_mb=budget,
            block_size=block_size,
            resident_bytes=retained + int(matrix.memory_usage(index=True, deep=True).sum()) + 160 * len(train),
            cancel=cancel,
        )
        checkpoint(cancel)
        # Publish only a complete fit; a second-stage admission failure keeps the old model usable.
        self._global_mean = overall_mean
        self._matrix, self._centered, self._item_means, self._sim = matrix, centered, item_means, similarities
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        if self._matrix is None or self._sim is None or self._item_means is None:
            raise RuntimeError("Call fit() before predict().")

        if user_id not in self._matrix.index or movie_id not in self._matrix.columns:
            return self._global_mean

        if movie_id not in self._sim.index:
            return self._global_mean

        user_row = self._matrix.loc[user_id]
        rated_items = user_row.dropna().index
        if len(rated_items) == 0:
            return float(self._item_means.get(movie_id, self._global_mean))

        sims = self._sim.loc[movie_id, rated_items].drop(labels=[movie_id], errors="ignore")
        sims = sims[sims > 0].sort_index().sort_values(ascending=False, kind="stable").head(self.k)

        if len(sims) == 0:
            return float(self._item_means.get(movie_id, self._global_mean))

        num = 0.0
        den = 0.0
        for other_movie, sim in sims.items():
            raw = user_row[other_movie]
            if np.isnan(raw):
                continue
            centered = raw - self._item_means[other_movie]
            num += sim * centered
            den += abs(sim)

        if den == 0:
            return float(self._item_means.get(movie_id, self._global_mean))

        item_mean = float(self._item_means.get(movie_id, self._global_mean))
        pred = item_mean + num / den
        return float(clip_predictions(np.array([pred]), *RATING_SCALE)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        preds = [
            self.predict(int(row.user_id), int(row.movie_id))
            for row in test.itertuples(index=False)
        ]
        return np.array(preds, dtype=float)

    def predict_for_user(self, user_id: int, movie_ids: list[int]) -> pd.Series:
        """Vectorized equivalent of calling `predict()` for each movie_id.

        Scores all candidate movies for one user in a single batch of numpy
        operations instead of one pandas `.loc` lookup per movie, which is
        the dominant cost when ranking ~1,600 candidates per request.
        """
        if self._matrix is None or self._sim is None or self._item_means is None:
            raise RuntimeError("Call fit() before predict_for_user().")

        result = pd.Series(self._global_mean, index=movie_ids, dtype=float)
        known_means = self._item_means.index.intersection(movie_ids)
        if len(known_means):
            result.loc[known_means] = self._item_means.loc[known_means].fillna(self._global_mean)

        if user_id not in self._matrix.index:
            return result

        user_row = self._matrix.loc[user_id]
        rated_items = user_row.dropna().index
        valid = [m for m in movie_ids if m in self._sim.index]
        if not valid or len(rated_items) == 0:
            return result

        sim_sub = self._sim.loc[valid, rated_items].to_numpy()
        sim_sub = np.where(sim_sub > 0, sim_sub, 0.0)

        if sim_sub.shape[1] > self.k:
            topk_idx = np.argsort(-sim_sub, axis=1, kind="stable")[:, : self.k]
            mask = np.zeros_like(sim_sub, dtype=bool)
            rows = np.arange(sim_sub.shape[0])[:, None]
            mask[rows, topk_idx] = True
            sim_sub = np.where(mask, sim_sub, 0.0)

        rated_values = user_row.loc[rated_items].to_numpy()
        rated_means = self._item_means.loc[rated_items].to_numpy()
        centered_rated = rated_values - rated_means

        num = sim_sub @ centered_rated
        den = sim_sub.sum(axis=1)

        item_means_valid = self._item_means.reindex(valid).fillna(self._global_mean).to_numpy()
        ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        preds = np.where(den > 0, item_means_valid + ratio, item_means_valid)
        result.loc[valid] = clip_predictions(preds, *RATING_SCALE)
        return result

    def similar_items(self, movie_id: int, n: int = 10) -> pd.Series:
        """Return top-n most similar movies by item–item similarity (training fit required)."""
        if self._sim is None:
            raise RuntimeError("Call fit() before similar_items().")
        if movie_id not in self._sim.index:
            return pd.Series(dtype=float)
        sims = self._sim.loc[movie_id].drop(labels=[movie_id], errors="ignore")
        sims = sims[sims > 0].sort_index().sort_values(ascending=False, kind="stable").head(n)
        return sims
