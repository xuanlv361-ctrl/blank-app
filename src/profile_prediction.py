"""Fold a visitor's explicit ratings into fixed benchmark models, without ID borrowing."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

from .config import RATING_SCALE
from .item_based_cf import ItemBasedCF
from .svd_model import SVDRecommender


def predict_profile(model, history, candidates, exclude_user=None):
    """Return scores, support counts, and per-item rating evidence (not probabilities)."""
    evidence = {}
    support = pd.Series(0, index=candidates, dtype=int)
    if isinstance(model, SVDRecommender):
        known = sorted(m for m in history if m in model._movie_idx)
        baseline = model._baseline
        scores = pd.Series(
            [baseline.global_mean_ + baseline.item_bias_.get(m, 0) for m in candidates], index=candidates
        )
        if not known:
            return scores.clip(*RATING_SCALE), support, evidence
        factors = model._svd.components_.T
        x = factors[[model._movie_idx[m] for m in known]]
        y = np.array([history[m] - baseline.global_mean_ - baseline.item_bias_.get(m, 0) for m in known])
        bias = float(y.sum() / (10 + len(y)))
        fold = Ridge(alpha=10.0, fit_intercept=False).fit(x, y - bias)
        valid = [m for m in candidates if m in model._movie_idx]
        if not valid:
            return scores.clip(*RATING_SCALE), support, evidence
        v = factors[[model._movie_idx[m] for m in valid]]
        scores.loc[valid] += bias + fold.predict(v)
        support.loc[valid] = len(known)
        similarities = cosine_similarity(v, x)
        # Exact signed rating contributions for the fixed ridge fold-in, including its bias.
        influence = v @ np.linalg.solve(x.T @ x + 10.0 * np.eye(x.shape[1]), x.T)
        contributions = (influence + (1 - influence.sum(axis=1))[:, None] / (10 + len(y))) * y
        for i, mid in enumerate(valid):
            evidence[mid] = [
                (known[j], history[known[j]], float(similarities[i, j]), float(contributions[i, j]))
                for j in np.argsort(-np.abs(contributions[i]), kind="stable")
            ][:3]
        return scores.clip(*RATING_SCALE), support, evidence

    matrix = model._matrix.sort_index()
    known = sorted(m for m in history if m in matrix.columns)
    valid = [m for m in candidates if m in matrix.columns]
    scores = pd.Series(model._global_mean, index=candidates, dtype=float)
    if not known or not valid:
        return scores, support, evidence
    values = np.array([history[m] for m in known])
    if isinstance(model, ItemBasedCF):
        sims = model._sim.loc[valid, known].to_numpy().copy()
        sims = np.maximum(sims, 0)
        for i, mid in enumerate(valid):
            if mid in known:
                sims[i, known.index(mid)] = 0
        keep = np.argsort(-sims, axis=1, kind="stable")[:, : model.k]
        mask = np.zeros_like(sims, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        sims *= mask
        residual = values - model._item_means.reindex(known).to_numpy()
        denominator = sims.sum(axis=1)
        scores.loc[valid] = model._item_means.reindex(valid).to_numpy() + np.divide(
            sims @ residual, denominator, out=np.zeros(len(valid)), where=denominator > 0
        )
        support.loc[valid] = (sims > 0).sum(axis=1)
        for i, mid in enumerate(valid):
            order = np.argsort(-np.abs(sims[i] * residual), kind="stable")
            evidence[mid] = [
                (known[j], history[known[j]], float(sims[i, j]), float(sims[i, j] * residual[j] / denominator[i]))
                for j in order
                if sims[i, j] > 0
            ][:3]
    else:
        observed = matrix[known].to_numpy()
        mask = np.isfinite(observed)
        common = mask.sum(axis=1)
        if model.metric == "pearson":
            other_mean = np.divide(np.nansum(observed, axis=1), common, out=np.zeros(len(matrix)), where=common > 0)
            own_mean = np.divide(mask @ values, common, out=np.zeros(len(matrix)), where=common > 0)
        else:
            other_mean = model._user_means.reindex(matrix.index).to_numpy()
            own_mean = np.full(len(matrix), values.mean())
        a = np.where(mask, observed - other_mean[:, None], 0)
        b = np.where(mask, values[None, :] - own_mean[:, None], 0)
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        valid_variance = denominator > 1e-12
        if model.metric == "pearson":
            tolerance = 32 * np.finfo(float).eps * common
            valid_variance &= (a * a).sum(axis=1) > tolerance * np.nansum(observed * observed, axis=1)
            valid_variance &= (b * b).sum(axis=1) > tolerance * (mask @ (values * values))
        similarities = np.divide((a * b).sum(axis=1), denominator, out=np.zeros(len(matrix)), where=valid_variance)
        if model.metric == "pearson":
            similarities[np.abs(similarities) <= 32 * np.finfo(float).eps * common] = 0
        similarities[common < max(2, model.min_common)] = 0
        if exclude_user is not None:
            similarities[matrix.index == exclude_user] = 0
        ratings = matrix[valid].to_numpy()
        weights = np.where(np.isfinite(ratings), np.maximum(similarities[:, None], 0), 0)
        top = np.argsort(-weights, axis=0, kind="stable")[: model.k]
        keep = np.zeros_like(weights, dtype=bool)
        np.put_along_axis(keep, top, True, axis=0)
        weights *= keep
        centered = np.nan_to_num(ratings - model._user_means.reindex(matrix.index).to_numpy()[:, None])
        total = weights.sum(axis=0)
        scores.loc[valid] = values.mean() + np.divide(
            (weights * centered).sum(axis=0), total, out=np.zeros(len(valid)), where=total > 0
        )
        support.loc[valid] = (weights > 0).sum(axis=0)
        for col, mid in enumerate(valid):
            shared = weights[:, col] @ mask
            order = np.argsort(-shared, kind="stable")
            evidence[mid] = [
                (known[j], history[known[j]], float(shared[j] / max(total[col], 1e-12))) for j in order if shared[j] > 0
            ][:3]
    return scores.clip(*RATING_SCALE), support, evidence
