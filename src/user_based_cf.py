"""User-based collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cancellation import checkpoint
from .cf_capacity import admit_cf_fit, model_array_bytes
from .config import K_NEIGHBORS, MIN_COMMON_RATINGS, RATING_SCALE, SIMILARITY_METRIC
from .metrics import clip_predictions
from .preprocessing import build_user_item_matrix, global_mean, mean_center_by_user
from .similarity import pairwise_similarity_matrix


class UserBasedCF:
    """Predict ratings from similar users' centered ratings."""

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
        self._user_means: pd.Series | None = None
        self._sim: pd.DataFrame | None = None
        self._global_mean: float = 0.0

    def fit(self, train: pd.DataFrame, *, cancel=None) -> UserBasedCF:
        """Fit the model on the training ratings and return self."""
        checkpoint(cancel)
        budget = getattr(self, "memory_budget_mb", None)
        block_size = getattr(self, "block_size", None)
        retained = model_array_bytes(self)
        admit_cf_fit(train, "user", memory_budget_mb=budget, block_size=block_size, resident_bytes=retained)
        overall_mean = global_mean(train)
        matrix = build_user_item_matrix(train, fill_value=np.nan)
        centered, user_means = mean_center_by_user(matrix)
        similarities = pairwise_similarity_matrix(
            centered,
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
        self._matrix, self._centered, self._user_means, self._sim = matrix, centered, user_means, similarities
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        if self._matrix is None or self._sim is None or self._user_means is None:
            raise RuntimeError("Call fit() before predict().")

        if user_id not in self._matrix.index or movie_id not in self._matrix.columns:
            return self._global_mean

        if user_id not in self._sim.index:
            return self._global_mean

        # Users who rated this movie
        col = self._matrix[movie_id]
        raters = col.dropna().index
        if len(raters) == 0:
            return float(self._user_means.get(user_id, self._global_mean))

        sims = self._sim.loc[user_id, raters].drop(labels=[user_id], errors="ignore")
        sims = sims[sims > 0].sort_index().sort_values(ascending=False, kind="stable").head(self.k)

        if len(sims) == 0:
            return float(self._user_means.get(user_id, self._global_mean))

        num = 0.0
        den = 0.0
        for other_id, sim in sims.items():
            raw = self._matrix.loc[other_id, movie_id]
            if np.isnan(raw):
                continue
            centered = raw - self._user_means[other_id]
            num += sim * centered
            den += abs(sim)

        if den == 0:
            return float(self._user_means.get(user_id, self._global_mean))

        user_mean = float(self._user_means.get(user_id, self._global_mean))
        pred = user_mean + num / den
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
        if self._matrix is None or self._sim is None or self._user_means is None:
            raise RuntimeError("Call fit() before predict_for_user().")

        result = pd.Series(self._global_mean, index=movie_ids, dtype=float)

        if user_id not in self._matrix.index or user_id not in self._sim.index:
            return result

        valid = [m for m in movie_ids if m in self._matrix.columns]
        if not valid:
            return result

        other_users = self._sim.index.drop(user_id, errors="ignore")
        sim_row = self._sim.loc[user_id, other_users].to_numpy()

        sub = self._matrix.loc[other_users, valid]
        notna_mask = sub.notna().to_numpy()
        values = sub.to_numpy()

        sims_matrix = np.where(notna_mask & (sim_row[:, None] > 0), sim_row[:, None], 0.0)

        if sims_matrix.shape[0] > self.k:
            topk_idx = np.argsort(-sims_matrix, axis=0, kind="stable")[: self.k, :]
            mask = np.zeros_like(sims_matrix, dtype=bool)
            cols = np.arange(sims_matrix.shape[1])[None, :]
            mask[topk_idx, cols] = True
            sims_matrix = np.where(mask, sims_matrix, 0.0)

        user_means_other = self._user_means.loc[other_users].to_numpy()[:, None]
        centered = np.where(notna_mask, values - user_means_other, 0.0)

        num = (sims_matrix * centered).sum(axis=0)
        den = np.abs(sims_matrix).sum(axis=0)

        fallback_value = float(self._user_means.get(user_id, self._global_mean))
        ratio = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        preds = np.where(den > 0, fallback_value + ratio, fallback_value)
        result.loc[valid] = clip_predictions(preds, *RATING_SCALE)
        return result
