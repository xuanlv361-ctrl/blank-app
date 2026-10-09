"""Matrix factorization with sklearn TruncatedSVD on bias residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .baselines import BiasBaseline
from .cancellation import checkpoint
from .config import RANDOM_STATE, RATING_SCALE, SVD_N_COMPONENTS, SVD_N_ITER
from .metrics import clip_predictions
from .preprocessing import global_mean


class SVDRecommender:
    """
    Latent-factor model fitted to residuals around a regularized bias baseline.

    Missing entries are omitted from the sparse residual matrix. The implicit
    zeros in CSR therefore mean "no residual from the bias estimate", not
    "missing rating equals zero".
    """

    def __init__(
        self,
        n_components: int = SVD_N_COMPONENTS,
        n_iter: int = SVD_N_ITER,
        random_state: int = RANDOM_STATE,
        baseline_reg: float = 1.0,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.baseline_reg = baseline_reg
        self._svd: TruncatedSVD | None = None
        self._user_factors: np.ndarray | None = None
        self._user_idx: dict[int, int] = {}
        self._movie_idx: dict[int, int] = {}
        self._global_mean: float = 0.0
        self._baseline: BiasBaseline | None = None

    def fit(self, train: pd.DataFrame, *, cancel=None) -> SVDRecommender:
        """Fit the model on the training ratings and return self."""
        checkpoint(cancel)
        self._global_mean = global_mean(train)
        self._baseline = BiasBaseline(n_epochs=20, reg=self.baseline_reg).fit(train, cancel=cancel)
        checkpoint(cancel)

        user_ids = sorted(train["user_id"].unique())
        movie_ids = sorted(train["movie_id"].unique())
        self._user_idx = {u: i for i, u in enumerate(user_ids)}
        self._movie_idx = {m: i for i, m in enumerate(movie_ids)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for index, row in enumerate(train.itertuples(index=False)):
            if index % 1024 == 0:
                checkpoint(cancel)
            user_id = int(row.user_id)
            movie_id = int(row.movie_id)
            residual = float(row.rating) - self._baseline.predict(user_id, movie_id)
            rows.append(self._user_idx[user_id])
            cols.append(self._movie_idx[movie_id])
            data.append(residual)

        n_users = len(user_ids)
        n_movies = len(movie_ids)
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_movies),
            dtype=np.float64,
        )

        n_comp = min(self.n_components, min(matrix.shape) - 1)
        if n_comp < 1:
            raise ValueError("Not enough ratings to fit TruncatedSVD.")

        self._svd = TruncatedSVD(
            n_components=n_comp,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._user_factors = self._svd.fit_transform(matrix)
        checkpoint(cancel)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return the predicted rating for the given (user_id, movie_id)."""
        if self._svd is None or self._user_factors is None or self._baseline is None:
            raise RuntimeError("Call fit() before predict().")

        baseline = self._baseline.predict(user_id, movie_id)
        if user_id not in self._user_idx or movie_id not in self._movie_idx:
            return baseline

        user_pos = self._user_idx[user_id]
        movie_pos = self._movie_idx[movie_id]
        latent = float(self._user_factors[user_pos] @ self._svd.components_[:, movie_pos])
        pred = baseline + latent
        return float(clip_predictions(np.array([pred]), *RATING_SCALE)[0])

    def predict_batch(self, test: pd.DataFrame) -> np.ndarray:
        """Return predicted ratings for every (user_id, movie_id) row in the input."""
        if self._svd is None or self._user_factors is None or self._baseline is None:
            raise RuntimeError("Call fit() before predict_batch().")

        preds = np.empty(len(test), dtype=float)
        components = self._svd.components_

        for i, row in enumerate(test.itertuples(index=False)):
            user_id = int(row.user_id)
            movie_id = int(row.movie_id)
            baseline = self._baseline.predict(user_id, movie_id)
            if user_id not in self._user_idx or movie_id not in self._movie_idx:
                preds[i] = baseline
                continue
            user_pos = self._user_idx[user_id]
            movie_pos = self._movie_idx[movie_id]
            latent = float(self._user_factors[user_pos] @ components[:, movie_pos])
            preds[i] = baseline + latent

        return clip_predictions(preds, *RATING_SCALE)

    def predict_for_user(self, user_id: int, movie_ids: list[int]) -> pd.Series:
        """Vectorized equivalent of calling `predict()` for each movie_id.

        Computes the bias baseline per candidate (cheap dict lookups) and
        adds the latent-factor contribution for all candidates at once via a
        single matrix-vector product, instead of looping `predict()`.
        """
        if self._svd is None or self._user_factors is None or self._baseline is None:
            raise RuntimeError("Call fit() before predict_for_user().")

        baselines = np.array(
            [self._baseline.predict(user_id, movie_id) for movie_id in movie_ids],
            dtype=float,
        )
        result = pd.Series(baselines, index=movie_ids, dtype=float)

        if user_id not in self._user_idx:
            return result

        valid = [m for m in movie_ids if m in self._movie_idx]
        if not valid:
            return result

        user_pos = self._user_idx[user_id]
        movie_positions = [self._movie_idx[m] for m in valid]
        latent = self._user_factors[user_pos] @ self._svd.components_[:, movie_positions]
        preds = result.loc[valid].to_numpy() + latent
        result.loc[valid] = clip_predictions(preds, *RATING_SCALE)
        return result
