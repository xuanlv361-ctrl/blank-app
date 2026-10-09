"""Similarity functions for collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from .cancellation import checkpoint
from .cf_capacity import admit_similarity, resolve_block_size


def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (NaN treated as 0 overlap)."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() == 0:
        return 0.0
    va, vb = a[mask], b[mask]
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        return 0.0
    return float(1.0 - cosine(va, vb))


def pearson_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation similarity between two rating vectors."""
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return 0.0
    va, vb = a[mask], b[mask]
    if np.ptp(va) == 0 or np.ptp(vb) == 0:
        return 0.0
    value = float(np.corrcoef(va, vb)[0, 1])
    return 0.0 if abs(value) <= 32 * np.finfo(float).eps * len(va) else value


def pairwise_similarity_matrix(
    matrix: pd.DataFrame,
    metric: str = "cosine",
    min_common: int = 1,
    *,
    memory_budget_mb: float | None = None,
    block_size: int | None = None,
    resident_bytes: int = 0,
    cancel=None,
) -> pd.DataFrame:
    """
    Compute similarity between all rows of a centered rating matrix.
    Only pairs with at least min_common co-rated entries get non-zero similarity.
    Exact overlap moments are tiled; only the compatible final result is square.
    This still needs O(n^2) final storage and O(n^2 * features) arithmetic.
    """
    if metric not in ("cosine", "pearson"):
        raise ValueError(f"Unknown metric: {metric}")

    size = resolve_block_size(block_size)
    n, features = matrix.shape
    admit_similarity(n, features, memory_budget_mb=memory_budget_mb, block_size=size, resident_bytes=resident_bytes)
    arr = matrix.to_numpy(dtype=np.float64)
    mask = np.isfinite(arr).astype(np.float64)
    filled = np.where(np.isfinite(arr), arr, 0.0)
    squared = filled * filled
    values = np.empty((n, n), dtype=np.float64)
    minimum = max(min_common, 2 if metric == "pearson" else 1)
    for start in range(0, n, size):
        checkpoint(cancel)
        left = slice(start, min(start + size, n))
        for other in range(start, n, size):
            checkpoint(cancel)
            right = slice(other, min(other + size, n))
            tile = _overlap_tile(filled, squared, mask, left, right, metric, minimum)
            values[left, right] = tile
            if other != start:
                values[right, left] = tile.T
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=matrix.index, columns=matrix.index)


def _overlap_tile(filled, squared, mask, left, right, metric, minimum):
    # Both directions are needed: each pair's norms use only its shared ratings.
    common = mask[left] @ mask[right].T
    numerator = filled[left] @ filled[right].T
    left_square = squared[left] @ mask[right].T
    right_square = mask[left] @ squared[right].T
    if metric == "pearson":
        left_sum = filled[left] @ mask[right].T
        right_sum = mask[left] @ filled[right].T
        divisor = np.maximum(common, 1)
        cross_mean = left_sum * right_sum / divisor
        left_mean_square = left_sum * left_sum / divisor
        right_mean_square = right_sum * right_sum / divisor
        # Moment subtraction loses precision for constant/near-constant overlaps.
        # Test each variance against its own rounding scale, not their product.
        tolerance = 32 * np.finfo(np.float64).eps * np.maximum(common, 1)
        left_tol = tolerance * np.maximum(left_square, left_mean_square)
        right_tol = tolerance * np.maximum(right_square, right_mean_square)
        cross_tol = tolerance * np.maximum(np.abs(numerator), np.abs(cross_mean))
        numerator -= cross_mean
        left_square -= left_mean_square
        right_square -= right_mean_square
        left_square[left_square <= left_tol] = 0
        right_square[right_square <= right_tol] = 0
        numerator[np.abs(numerator) <= cross_tol] = 0
    np.maximum(left_square, 0, out=left_square)
    np.maximum(right_square, 0, out=right_square)
    left_square *= right_square
    np.sqrt(left_square, out=left_square)
    valid = (common >= minimum) & (left_square > 1e-12)
    np.divide(numerator, left_square, out=numerator, where=valid)
    numerator[~valid] = 0
    np.clip(numerator, -1, 1, out=numerator)
    return numerator
