"""Explicit, reproducible score alignment and reranking policies.

Percentiles align ordering scales, not probabilities or predicted star ratings.
The previous raw blend remains an explicit experimental baseline.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingPolicy:
    alignment: str = "legacy"
    cf_weight: float = 0.85
    diversity_weight: float = 0.10
    freshness_weight: float = 0.0
    reference_year: int = 2026

    def validate(self):
        if self.alignment not in {"legacy", "percentile"}:
            raise ValueError("Unknown score alignment")
        for value in (self.cf_weight, self.diversity_weight, self.freshness_weight):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Ranking weights must be finite values in [0, 1]")
        if not isinstance(self.reference_year, int) or not 1900 <= self.reference_year <= 2200:
            raise ValueError("Invalid ranking reference year")
        return self

    def as_dict(self):
        return asdict(self.validate())


def percentile_scale(values, *, positive_only=False):
    """Deterministic midranks; ties stay tied and no-match content remains zero."""
    values = pd.Series(values, dtype=float)
    output = pd.Series(0.0, index=values.index)
    eligible = values.notna() & np.isfinite(values)
    if positive_only:
        eligible &= values > 0
    subset = values.loc[eligible]
    if len(subset):
        output.loc[eligible] = (subset.rank(method="average") - 0.5) / len(subset)
    return output


def align_scores(frame, policy, has_preferences=True):
    """Use one content distribution for old/new titles; never use TMDB votes as CF."""
    policy.validate()
    result = frame.copy()
    supported = result["Predicted Rating"].notna()
    content = result["Content Match"]
    if policy.alignment == "percentile" and has_preferences:
        content = percentile_scale(content, positive_only=True)
        cf = percentile_scale(result["Predicted Rating"])
        scores = content.copy()
        scores.loc[supported] = policy.cf_weight * cf[supported] + (1 - policy.cf_weight) * content[supported]
    else:
        scores = result["Recommendation Score"].copy()
        scores.loc[supported] = (
            policy.cf_weight * (result.loc[supported, "Predicted Rating"] - 0.5) / 4.5
            + (1 - policy.cf_weight) * content[supported]
        )
    years = pd.to_numeric(result["Release Year"], errors="coerce")
    recency = (1 - (policy.reference_year - years) / 5).clip(0, 1)
    recency = recency.where(years <= policy.reference_year, 0).fillna(0)
    result["Recommendation Score"] = (1 - policy.freshness_weight) * scores + policy.freshness_weight * recency
    result["Content Percentile"] = percentile_scale(result["Content Match"], positive_only=True)
    result["Freshness"] = recency
    result["Ranking Policy"] = policy.alignment
    return result


POLICIES = {
    "legacy-85": RankingPolicy(),
    "percentile-50": RankingPolicy("percentile", 0.50),
    "percentile-70": RankingPolicy("percentile", 0.70),
    "percentile-85": RankingPolicy("percentile", 0.85),
    "percentile-70-fresh": RankingPolicy("percentile", 0.70, freshness_weight=0.05),
    "percentile-70-diverse": RankingPolicy("percentile", 0.70, diversity_weight=0.25),
}
