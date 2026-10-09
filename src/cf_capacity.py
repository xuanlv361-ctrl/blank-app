"""Conservative admission for exact dense CF, not an out-of-core implementation.

CF_MEMORY_BUDGET_MB (default 512, decimal MB) bounds the estimated working set
of one operation. Half of currently available RAM is an additional ceiling.
CF_SIMILARITY_BLOCK_SIZE (default 128) bounds overlap-moment tiles. Smaller
tiles reduce scratch space, but the retained float64 similarity is still
O(users**2) or O(items**2), and the rating matrices are still dense. Estimates
include conversion/pivot scratch and headroom; they are not an OS reservation
or a limit on aggregate concurrent models, pandas, BLAS, or serving memory.
"""

from __future__ import annotations

import math
import os

import pandas as pd
import psutil

MB = 1_000_000
DEFAULT_MEMORY_BUDGET_MB = 512.0
DEFAULT_BLOCK_SIZE = 128


class CFCapacityError(MemoryError):
    """A dense CF operation was rejected before its large allocations."""


def resolve_block_size(block_size: int | None = None) -> int:
    value = block_size if block_size is not None else os.environ.get("CF_SIMILARITY_BLOCK_SIZE", DEFAULT_BLOCK_SIZE)
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CF similarity block size must be a positive integer") from exc
    if result < 1 or isinstance(value, bool) or (not isinstance(value, str) and result != value):
        raise ValueError("CF similarity block size must be a positive integer")
    return result


def memory_budget_bytes(memory_budget_mb: float | None = None) -> int:
    value = memory_budget_mb if memory_budget_mb is not None else os.environ.get(
        "CF_MEMORY_BUDGET_MB", DEFAULT_MEMORY_BUDGET_MB
    )
    try:
        configured = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CF memory budget must be finite and positive (MB)") from exc
    if not math.isfinite(configured) or configured <= 0:
        raise ValueError("CF memory budget must be finite and positive (MB)")
    available_limit = int(psutil.virtual_memory().available) // 2
    return min(int(min(configured, available_limit / MB) * MB), available_limit)


def rating_dimensions(ratings: pd.DataFrame) -> tuple[int, int]:
    """Upper bound pivot dimensions, including unobserved categorical labels."""
    dimensions = []
    for column in ("user_id", "movie_id"):
        values = ratings[column]
        count = len(values.cat.categories) if isinstance(values.dtype, pd.CategoricalDtype) else values.nunique()
        dimensions.append(int(count))
    return dimensions[0], dimensions[1]


def model_array_bytes(model) -> int:
    """Retained numeric/pandas payload, including indexes; not total Python RSS."""
    total = 0
    seen = set()
    for value in vars(model).values():
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, pd.DataFrame):
            total += int(value.memory_usage(index=True, deep=True).sum())
            total += int(value.columns.memory_usage(deep=True))
        elif isinstance(value, pd.Series):
            total += int(value.memory_usage(index=True, deep=True))
        elif hasattr(value, "nbytes"):
            total += int(value.nbytes)
    return total


def estimate_similarity_bytes(rows: int, features: int, block_size: int, resident_bytes: int = 0) -> int:
    """Input, conversion, mask/filled/squared, final square, and <=12 tile arrays.

    Use Python integers so large, sparse input dimensions cannot overflow the
    estimate. A 50% margin plus 16 MB covers allocator/pandas/BLAS bookkeeping.
    The caller supplies other retained arrays and long-form rating scratch.
    """
    rows, features = int(rows), int(features)
    tile = min(rows, int(block_size))
    payload = 8 * (5 * rows * features + rows * rows + 12 * tile * tile)
    payload += 256 * (rows + features) + int(resident_bytes)
    return (3 * payload + 1) // 2 + 16 * MB


def _admit(estimated: int, memory_budget_mb: float | None, stage: str, dimensions: str) -> int:
    budget = memory_budget_bytes(memory_budget_mb)
    if estimated > budget:
        raise CFCapacityError(
            f"{stage} rejected: estimated {estimated / MB:.3f} MB exceeds budget {budget / MB:.3f} MB "
            f"({dimensions}). Dense CF retains an O(n^2) similarity matrix. Reduce dimensions or adjust "
            "CF_MEMORY_BUDGET_MB only with sufficient free RAM; available-memory ceiling is 50%."
        )
    return estimated


def admit_similarity(
    rows: int,
    features: int,
    *,
    memory_budget_mb: float | None = None,
    block_size: int | None = None,
    resident_bytes: int = 0,
) -> int:
    estimate = estimate_similarity_bytes(rows, features, resolve_block_size(block_size), resident_bytes)
    return _admit(estimate, memory_budget_mb, "CF before similarity allocation", f"rows={rows}, features={features}")


def admit_cf_fit(
    ratings: pd.DataFrame,
    axis: str,
    *,
    memory_budget_mb: float | None = None,
    block_size: int | None = None,
    resident_bytes: int = 0,
    stage: str = "CF before pivot",
) -> int:
    if axis not in {"item", "user"}:
        raise ValueError("CF axis must be item or user")
    users, items = rating_dimensions(ratings)
    rows, features = (items, users) if axis == "item" else (users, items)
    # One additional retained raw matrix plus long-form groupby/pivot scratch.
    retained = int(resident_bytes) + 8 * users * items + 160 * len(ratings)
    estimate = estimate_similarity_bytes(rows, features, resolve_block_size(block_size), retained)
    return _admit(estimate, memory_budget_mb, stage, f"users={users}, items={items}, similarity={axis}")
