"""Evaluation metrics for rating prediction and top-k recommendation."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the root-mean-squared error between true and predicted ratings."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error between true and predicted ratings."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def clip_predictions(
    preds: np.ndarray,
    low: float = 1.0,
    high: float = 5.0,
) -> np.ndarray:
    """Clip predicted ratings into the valid [low, high] rating range."""
    return np.clip(preds, low, high)


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Share of the top-k recommendations that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Share of relevant items recovered by the top-k recommendations."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Whether at least one relevant item appears in the top-k list."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0
    return float(any(item_id in relevant for item_id in recommended[:k]))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@K for one user's recommendation list."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg else 0.0


def catalog_coverage(recommendations: list[list[int]], all_items: set[int]) -> float:
    """Fraction of catalog items that appear in at least one recommendation list."""
    if not all_items:
        return 0.0
    recommended_items = {
        item_id
        for user_recommendations in recommendations
        for item_id in user_recommendations
    }
    return len(recommended_items & all_items) / len(all_items)


def evaluate_topk(
    recommendations: dict[int, list[int]],
    relevant_items: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    """Average top-k metrics across users with at least one relevant item."""
    users = [user_id for user_id, items in relevant_items.items() if items]
    if not users:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    precision = []
    recall = []
    hit_rate = []
    ndcg = []
    for user_id in users:
        recommended = recommendations.get(user_id, [])
        relevant = relevant_items[user_id]
        precision.append(precision_at_k(recommended, relevant, k))
        recall.append(recall_at_k(recommended, relevant, k))
        hit_rate.append(hit_rate_at_k(recommended, relevant, k))
        ndcg.append(ndcg_at_k(recommended, relevant, k))

    return {
        "precision_at_k": float(np.mean(precision)),
        "recall_at_k": float(np.mean(recall)),
        "hit_rate_at_k": float(np.mean(hit_rate)),
        "ndcg_at_k": float(np.mean(ndcg)),
    }
