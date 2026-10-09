"""Shared, feedback-aware serving with factual evidence and metadata-only discovery."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .content_features import CONTENT_VERSION, TFIDF_OPTIONS, ContentFeatures
from .model_store import cache_key, checkpoint, fingerprint, get_model
from .movie_identity import CatalogIdentity, IdentityConflict
from .personal_feedback import Profile
from .profile_prediction import predict_profile
from .ranking_policy import RankingPolicy, align_scores
from .recsys_service import genre_columns

ALGORITHMS = ["Item-based CF", "User-based CF", "SVD"]
SERVING_VERSION = "feedback-v5-deterministic-profiles"


def catalog_metadata(movies, discovery=()):
    """Canonical text inputs shared by serving and experiment-cache identity."""
    genres = sorted(genre_columns(movies))
    identity = CatalogIdentity(movies.to_dict("records"))
    aliases = {key for key in identity.primary.values() if key.startswith("tmdb:")}
    rows = []
    for movie in movies.sort_values("movie_id").to_dict("records"):
        mid = int(movie["movie_id"])
        rows.append(
            {
                "Key": f"catalog:{mid}",
                "Movie ID": mid,
                "Title": movie["title"],
                "Genres": ", ".join(g for g in genres if pd.notna(movie.get(g)) and movie.get(g) == 1),
                "Release Year": movie.get("release_year"),
                "Overview": movie.get("overview"),
                "Source": "Catalog",
                "TMDB Score": np.nan,
            }
        )
    for movie in sorted(discovery, key=lambda row: int(row["tmdb_id"])):
        key = f"tmdb:{int(movie['tmdb_id'])}"
        if key in aliases:
            continue
        rows.append(
            {
                "Key": key,
                "Movie ID": -int(movie["tmdb_id"]),
                "Title": movie["title"],
                "Genres": ", ".join(sorted(set(movie["genres"]))),
                "Release Year": int(movie["release_date"][:4]),
                "Overview": movie.get("overview"),
                "Source": "TMDB new release",
                "TMDB Score": movie.get("tmdb_vote_average", np.nan),
            }
        )
    return pd.DataFrame(rows).set_index("Key", drop=False), identity


def content_documents(catalog):
    text = catalog.Title.fillna("") + " " + catalog.Overview.fillna("") + " " + (catalog.Genres.fillna("") + " ") * 4
    decades = pd.to_numeric(catalog["Release Year"], errors="coerce").floordiv(10).mul(10)
    return text + " " + decades.map(lambda year: f"era{int(year)}s" if pd.notna(year) else "")


def content_input_signature(movies, discovery=()):
    catalog, _ = catalog_metadata(movies, discovery)
    return cache_key(CONTENT_VERSION, TFIDF_OPTIONS, catalog.Key.tolist(), content_documents(catalog).tolist())


class RecommendationRuntime:
    def __init__(self, ratings, movies, discovery=(), *, cache_dir=None, content_backend="tfidf", ranking_policy=None):
        self.ratings = ratings.copy(deep=True)
        self.movies = movies.copy(deep=True)
        self.cache_dir = cache_dir
        self.generation = 0
        self.ranking_policy = (ranking_policy or RankingPolicy()).validate()
        self.content_backend = content_backend
        counts = ratings.groupby("movie_id").rating.agg(["mean", "count"])
        self.catalog, identity = catalog_metadata(movies, discovery)
        self.identity = identity
        self.aliases = {key: f"catalog:{mid}" for mid, key in identity.primary.items() if key.startswith("tmdb:")}
        self.catalog["Average Rating"] = self.catalog["Movie ID"].map(counts["mean"])
        self.catalog["Ratings"] = self.catalog["Movie ID"].map(counts["count"]).fillna(0).astype(int)
        self.keys = self.catalog.index.tolist()
        self.positions = {key: i for i, key in enumerate(self.keys)}
        text = content_documents(self.catalog)
        self.content_index = ContentFeatures(text, content_backend)
        self.features = self.content_index.matrix
        self.version = cache_key(
            SERVING_VERSION,
            fingerprint(ratings),
            fingerprint(self.catalog.reset_index(drop=True)),
            self.content_index.version,
            sorted(identity.sources.items()),
            sorted(identity.primary.items()),
            sorted(identity.conflicts),
            self.ranking_policy.as_dict(),
        )
        self._results = OrderedDict()
        self._lock = threading.RLock()

    def canonical(self, profile):
        ratings = {}
        for key, value in sorted(profile.ratings.items()):
            target = self.identity.project(key)
            if target in ratings and ratings[target] != value:
                raise IdentityConflict("Conflicting ratings share a verified movie identity")
            ratings[target] = value
        return Profile(
            dict(sorted(ratings.items())),
            {self.identity.project(k) for k in profile.blocked},
            sorted(set(profile.genres)),
            profile.revision,
        )

    def prewarm(self, progress=None, cancel=None):
        for algorithm in ALGORITHMS:
            get_model(
                self.ratings,
                algorithm,
                generation=self.generation,
                progress=progress,
                cancel=cancel,
                cache_dir=self.cache_dir,
            )

    def rebuild(self):
        with self._lock:
            self.generation += 1
            self._results.clear()

    def predict_ratings(
        self,
        profile,
        movie_ids,
        algorithm="Item-based CF",
        k=20,
        metric="cosine",
        factors=50,
        *,
        sample_id=None,
        progress=None,
        cancel=None,
    ):
        """Shared serving predictions; unsupported items have no displayed rating.

        Evaluation uses this same path rather than a dataset user's fitted vector.
        Content scores are deliberately never converted into rating predictions.
        """
        if algorithm not in ALGORITHMS:
            raise ValueError("Invalid recommendation algorithm")
        checkpoint(cancel)
        profile = self.canonical(profile)
        history = {
            int(key.split(":")[1]): value
            for key, value in profile.ratings.items()
            if key.startswith("catalog:") and key in self.positions and key not in profile.blocked
        }
        scores = pd.Series(np.nan, index=movie_ids, dtype=float)
        support = pd.Series(0, index=movie_ids, dtype=int)
        if not history or not movie_ids:
            return scores, support, {}, {"cache": "not needed"}
        model, info = get_model(
            self.ratings,
            algorithm,
            k,
            metric,
            factors,
            generation=self.generation,
            progress=progress,
            cancel=cancel,
            cache_dir=self.cache_dir,
        )
        checkpoint(cancel)
        scores, support, evidence = predict_profile(model, history, movie_ids, exclude_user=sample_id)
        return scores.where(support > 0), support, evidence, info

    def recommend(
        self,
        profile,
        algorithm="Item-based CF",
        top_n=10,
        k=20,
        metric="cosine",
        factors=50,
        include_new=True,
        sample_id=None,
        progress=None,
        cancel=None,
        new_only=False,
        ranking_policy=None,
    ):
        if algorithm not in ALGORITHMS or not 1 <= top_n <= 50:
            raise ValueError("Invalid recommendation settings")
        checkpoint(cancel)
        policy = (ranking_policy or self.ranking_policy).validate()
        profile = self.canonical(profile)
        token = cache_key(
            self.version,
            self.generation,
            sorted(profile.ratings.items()),
            sorted(profile.blocked),
            sorted(profile.genres),
            profile.revision,
            algorithm,
            top_n,
            k,
            metric,
            factors,
            include_new,
            sample_id,
            new_only,
            policy.as_dict(),
        )
        with self._lock:
            if token in self._results:
                result = self._results[token].copy(deep=True)
                result.attrs = {**result.attrs, "cache": "result", "seconds": 0.0}
                return result
        start = time.perf_counter()
        excluded = set(profile.ratings) | profile.blocked
        candidates = [
            key
            for key in self.keys
            if key not in excluded
            and (include_new or self.catalog.loc[key, "Source"] == "Catalog")
            and (not new_only or self.catalog.loc[key, "Source"] == "TMDB new release")
        ]
        if not candidates:
            return pd.DataFrame()
        catalog_ids = [int(key.split(":")[1]) for key in candidates if key.startswith("catalog:")]
        scores, support, evidence, model_info = self.predict_ratings(
            profile,
            catalog_ids,
            algorithm,
            k,
            metric,
            factors,
            sample_id=sample_id,
            progress=progress,
            cancel=cancel,
        )
        if progress:
            progress("Matching preferences and diversifying results")
        content = np.zeros(len(self.keys))
        known_keys = [key for key in profile.ratings if key in self.positions and key not in profile.blocked]
        positive = [key for key in known_keys if profile.ratings[key] >= 4]
        positive_sims = np.zeros((len(self.keys), 0))
        if known_keys:
            similarities = cosine_similarity(self.features, self.features[[self.positions[key] for key in known_keys]])
            weights = np.array([(profile.ratings[key] - 3) / 2 for key in known_keys])
            content = similarities @ weights / max(1, len(known_keys))
            positive_sims = similarities[:, [known_keys.index(key) for key in positive]]
        if profile.genres:
            preference = self.content_index.transform([" ".join(profile.genres)])
            content += 0.35 * cosine_similarity(preference, self.features).ravel()
        has_preferences = bool(known_keys or profile.genres)
        rows = []
        for key in candidates:
            checkpoint(cancel)
            row = self.catalog.loc[key].to_dict()
            mid = int(row["Movie ID"])
            count = int(support.get(mid, 0)) if key.startswith("catalog:") else 0
            pred = float(scores.get(mid, np.nan)) if count else np.nan
            match = float(np.clip(content[self.positions[key]], 0, 1))
            prior = (
                row["Ratings"] * (row["Average Rating"] if pd.notna(row["Average Rating"]) else 3.5) + 50 * 3.5
            ) / (row["Ratings"] + 50)
            if pd.notna(pred):
                ranking = 0.85 * (pred - 0.5) / 4.5 + 0.15 * match
                kind = "Personalized rating / 5"
                pieces = []
                for source, rating, sim, *contribution in evidence.get(mid, []):
                    source_key = f"catalog:{source}"
                    if source_key in self.catalog.index:
                        label = (
                            "latent similarity"
                            if algorithm == "SVD"
                            else ("shared-neighbor support" if algorithm == "User-based CF" else "rating similarity")
                        )
                        pieces.append(
                            f"{self.catalog.loc[source_key, 'Title']} (your rating {rating:g}/5; {label} {sim:.2f})"
                            + (f"; contribution {contribution[0]:+.3f} rating points" if contribution else "")
                        )
                basis = (
                    f"{count} supporting neighbors"
                    if algorithm != "SVD"
                    else f"{count} ratings used for latent-profile fitting"
                )
                reason = basis + (
                    ". Evidence: " + "; ".join(pieces) if pieces else ". No item-level evidence available."
                )
            elif has_preferences:
                ranking = match
                kind = "Content match (not a rating)"
                if positive:
                    sims = positive_sims[self.positions[key]]
                    best = [(positive[i], float(sims[i])) for i in np.argsort(-sims)[:2] if sims[i] > 0]
                    reason = "Synopsis/genre match: " + "; ".join(
                        f"{self.catalog.loc[p, 'Title']} (your rating {profile.ratings[p]:g}/5; "
                        f"content similarity {sim:.2f})"
                        for p, sim in best
                    )
                    if not best:
                        reason = "No positive history overlap; low-support exploration."
                else:
                    reason = (
                        "Matches selected genres: " + ", ".join(profile.genres)
                        if profile.genres
                        else "No positive preference evidence yet."
                    )
            else:
                ranking = (prior - 0.5) / 4.5 if row["Source"] == "Catalog" else 0
                kind = "Popularity fallback (not personalized)"
                reason = "Popularity-shrunk community rating; no personal ratings yet."
            row.update(
                {
                    "Predicted Rating": pred,
                    "Recommendation Score": ranking,
                    "Content Match": match,
                    "Support Count": count,
                    "Support Strength": "High" if count >= 10 else ("Medium" if count >= 3 else "Low"),
                    "Score Type": kind,
                    "Reason": reason,
                }
            )
            rows.append(row)
        result = align_scores(pd.DataFrame(rows), policy, has_preferences)
        result = result.sort_values(["Recommendation Score", "Key"], ascending=[False, True])
        # Greedy MMR over a bounded pool; score labels remain distinct from the display order.
        pool = result.head(max(50, top_n * 5)).copy()
        selected = []
        while len(selected) < min(top_n, len(result)) and not pool.empty:
            checkpoint(cancel)
            penalty = np.zeros(len(pool))
            if selected:
                penalty = cosine_similarity(
                    self.features[[self.positions[p] for p in pool.Key]],
                    self.features[[self.positions[p] for p in selected]],
                ).max(axis=1)
            objective = (1 - policy.diversity_weight) * pool["Recommendation Score"].to_numpy()
            chosen = pool.iloc[int(np.argmax(objective - policy.diversity_weight * penalty))]["Key"]
            selected.append(chosen)
            pool = pool[pool.Key != chosen]
        result = result.set_index("Key", drop=False).loc[selected].reset_index(drop=True)
        result.insert(0, "Rank", range(1, len(result) + 1))
        result.attrs = {
            "version": token,
            "seconds": time.perf_counter() - start,
            "cache": model_info["cache"],
            "dataset": self.version,
            "feedback_revision": profile.revision,
            "ranking_policy": policy.as_dict(),
            "content_backend": self.content_backend,
        }
        with self._lock:
            self._results[token] = result.copy(deep=True)
            while len(self._results) > 64:
                self._results.popitem(last=False)
        return result
