"""Account-owned feedback, separate from immutable benchmark interactions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .account_storage import ACCOUNT_DB_PATH as DB_PATH
from .account_storage import get_connection, require_active
from .movie_identity import IdentityConflict, alias_map, bind_reference, catalog_for, validate_stable_key

__all__ = [
    "DB_PATH",
    "Profile",
    "ensure_feedback_schema",
    "list_personal_history",
    "read_profile",
    "sample_profile",
    "save_feedback",
    "validate_feedback",
]


@dataclass
class Profile:
    ratings: dict[str, float] = field(default_factory=dict)
    blocked: set[str] = field(default_factory=set)
    genres: list[str] = field(default_factory=list)
    revision: int = 0


def ensure_feedback_schema(db_path=None):
    get_connection(db_path).close()


def validate_feedback(item_key, rating):
    if not isinstance(item_key, str):
        raise ValueError("Invalid movie identifier")
    if not re.fullmatch(r"catalog:[1-9][0-9]*", item_key):
        validate_stable_key(item_key)
    if rating is not None and not (0.5 <= float(rating) <= 5 and float(rating) * 2 % 1 == 0):
        raise ValueError("Ratings must be half-star values from 0.5 to 5")


def save_feedback(account_id, item_key, *, rating=None, blocked=None, db_path=None, catalog_db_path=None):
    validate_feedback(item_key, rating)
    if blocked is not None and blocked not in (True, False, 0, 1):
        raise ValueError("Blocked must be a boolean")
    reference = catalog_for(db_path, catalog_db_path).reference(item_key)
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            require_active(conn, account_id)
            item_key = bind_reference(conn, reference)
            old = conn.execute(
                "SELECT rating, blocked FROM account_feedback WHERE account_id=? AND item_key=?", (account_id, item_key)
            ).fetchone()
            value = float(rating) if rating is not None else (old["rating"] if old else None)
            hidden = int(blocked) if blocked is not None else (old["blocked"] if old else 0)
            conn.execute(
                """INSERT INTO account_feedback VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_id,item_key) DO UPDATE SET rating=excluded.rating,
                blocked=excluded.blocked, updated_at=excluded.updated_at""",
                (account_id, item_key, value, hidden, datetime.now(UTC).isoformat()),
            )
            conn.execute(
                """INSERT INTO feedback_versions VALUES (?, 1)
                ON CONFLICT(account_id) DO UPDATE SET revision=revision+1""",
                (account_id,),
            )
    finally:
        conn.close()


def _feedback_rows(account_id, db_path):
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT f.item_key,f.rating,f.blocked,f.updated_at,r.title FROM account_feedback f "
            "JOIN item_refs r USING(item_key) WHERE account_id=? ORDER BY updated_at,f.item_key",
            (account_id,),
        ).fetchall()
        version = conn.execute("SELECT revision FROM feedback_versions WHERE account_id=?", (account_id,)).fetchone()
        aliases = alias_map(conn)
    finally:
        conn.close()
    return rows, int(version[0]) if version else 0, aliases


def read_profile(account_id, genres=(), db_path=None, *, project=True, catalog_db_path=None):
    """Default runtime-compatible projection; project=False returns durable keys."""
    rows, revision, aliases = _feedback_rows(account_id, db_path)
    catalog = catalog_for(db_path, catalog_db_path) if project else None
    projected = {}
    for row in rows:
        key = catalog.project(row["item_key"], aliases[row["item_key"]]) if project else row["item_key"]
        if key in projected:
            raise IdentityConflict("Multiple stored feedback rows project to one movie; reconciliation required")
        projected[key] = row
    return Profile(
        {key: float(row["rating"]) for key, row in projected.items() if row["rating"] is not None},
        {key for key, row in projected.items() if row["blocked"]},
        list(genres),
        revision,
    )


def list_personal_history(account_id, db_path=None, *, catalog_db_path=None, off_catalog_only=False):
    """List durable feedback including items absent from the selected catalog.

    Returned rows are private account data; callers must not log them.
    Availability refers to the benchmark catalog, not the discovery feed.
    """
    rows, _, aliases = _feedback_rows(account_id, db_path)
    catalog = catalog_for(db_path, catalog_db_path)
    result = []
    for row in rows:
        projected = catalog.project(row["item_key"], aliases[row["item_key"]])
        available = projected.startswith("catalog:")
        if not off_catalog_only or not available:
            result.append({**dict(row), "available": available, "catalog_key": projected if available else None})
    return result


def sample_profile(ratings, user_id):
    rows = ratings.loc[ratings.user_id == user_id]
    return Profile({f"catalog:{int(r.movie_id)}": float(r.rating) for r in rows.itertuples()})
