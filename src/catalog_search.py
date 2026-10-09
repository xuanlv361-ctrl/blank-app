"""Literal catalog filtering shared by the desktop and personal web views."""

from __future__ import annotations

import re

import pandas as pd


def catalog_genres(catalog: pd.DataFrame) -> list[str]:
    """Return the available genre tokens, independently of the current search."""
    return sorted(
        {
            genre.strip()
            for value in catalog["Genres"].dropna().astype(str)
            for genre in value.split(",")
            if genre.strip()
        }
    )


def filter_catalog(catalog: pd.DataFrame, query: str, genre: str = "All", *, title_aliases=None) -> pd.DataFrame:
    """Match exact signed IDs or literal title/genre text, then an optional genre.

    Numeric queries never fall back to title matching. The returned copy retains
    the caller's row order, index and columns; ranking and pagination stay in UI code.
    """
    query = query.strip()
    genres = catalog["Genres"].astype("string").fillna("").str.casefold()
    matches = pd.Series(True, index=catalog.index, dtype=bool)
    if query:
        if re.fullmatch(r"[+-]?[0-9]+", query):
            try:
                movie_id = int(query)
            except ValueError:
                return catalog.iloc[:0].copy()
            matches = catalog["Movie ID"].eq(movie_id).fillna(False)
        else:
            needle = query.casefold()
            titles = catalog["Title"].astype("string").str.casefold()
            matches = titles.str.contains(needle, regex=False, na=False) | genres.str.contains(
                needle, regex=False, na=False
            )
            if title_aliases is not None:
                matches |= (
                    title_aliases.reindex(catalog.index)
                    .astype("string")
                    .str.casefold()
                    .str.contains(needle, regex=False, na=False)
                )
    selected = genre.strip().casefold()
    if selected and selected != "all":
        matches &= genres.map(lambda value: selected in [part.strip() for part in value.split(",")])
    return catalog.loc[matches].copy()
