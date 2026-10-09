"""Dataset-independent account storage. Explicit paths never inspect live data."""

from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_DB_PATH = ROOT / "data" / "private" / "accounts.sqlite3"
SCHEMA_VERSION = 1


class MigrationRequired(RuntimeError):
    pass


def account_path(db_path=None):
    if db_path is not None:
        return Path(db_path).resolve()
    path = Path(os.environ.get("FILMTRACE_ACCOUNT_DB", str(ACCOUNT_DB_PATH)))
    if not path.is_absolute():
        raise ValueError("FILMTRACE_ACCOUNT_DB must be an absolute path")
    path = path.resolve()
    if path.is_relative_to((ROOT / "data" / "processed").resolve()):
        raise ValueError("Account storage must be outside processed datasets")
    return path


def connect(path, *, readonly=False):
    path = Path(path).resolve()
    if readonly:
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def discover_legacy_sources(root=None):
    base = (Path(root or ROOT) / "data" / "processed").resolve()
    paths = [base / "app.db", base / "letterboxd-2025" / "app.db"]
    paths.extend(sorted((base / "letterboxd-2025").glob("*/app.db")))
    return [p.resolve() for p in paths if p.is_file() and p.resolve().is_relative_to(base)]


_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS account_storage_meta (version INTEGER NOT NULL)",
    """CREATE TABLE IF NOT EXISTS accounts (
        account_id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, movielens_user_id INTEGER,
        is_active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS item_refs (
        item_key TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '')""",
    """CREATE TABLE IF NOT EXISTS item_aliases (
        alias_key TEXT PRIMARY KEY, item_key TEXT NOT NULL REFERENCES item_refs(item_key))""",
    """CREATE TABLE IF NOT EXISTS account_feedback (
        account_id INTEGER NOT NULL REFERENCES accounts(account_id),
        item_key TEXT NOT NULL REFERENCES item_refs(item_key), rating REAL,
        blocked INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
        PRIMARY KEY(account_id,item_key),
        CHECK(rating IS NULL OR (rating BETWEEN 0.5 AND 5 AND rating*2=CAST(rating*2 AS INTEGER))),
        CHECK(blocked IN (0,1)))""",
    """CREATE TABLE IF NOT EXISTS feedback_versions (
        account_id INTEGER PRIMARY KEY REFERENCES accounts(account_id), revision INTEGER NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS favorites (
        account_id INTEGER NOT NULL REFERENCES accounts(account_id),
        item_key TEXT NOT NULL REFERENCES item_refs(item_key), created_at TEXT NOT NULL,
        PRIMARY KEY(account_id,item_key))""",
    """CREATE TABLE IF NOT EXISTS user_preferences (
        account_id INTEGER PRIMARY KEY REFERENCES accounts(account_id), genres TEXT NOT NULL,
        seed_item_keys TEXT NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS user_profiles (
        account_id INTEGER PRIMARY KEY REFERENCES accounts(account_id), top_genres TEXT NOT NULL,
        year_range TEXT NOT NULL, preferred_min_rating REAL NOT NULL, updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS account_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, operation TEXT NOT NULL,
        administrator TEXT NOT NULL, target TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS migration_sources (
        source_path TEXT PRIMARY KEY, content_digest TEXT NOT NULL, completed_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS catalog_provenance (
        catalog_path TEXT PRIMARY KEY, namespace TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS migration_accounts (
        source_path TEXT NOT NULL REFERENCES migration_sources(source_path) DEFERRABLE INITIALLY DEFERRED,
        legacy_id INTEGER NOT NULL, account_id INTEGER NOT NULL REFERENCES accounts(account_id),
        PRIMARY KEY(source_path,legacy_id))""",
    """CREATE TABLE IF NOT EXISTS legacy_recovery (
        source_path TEXT NOT NULL, table_name TEXT NOT NULL, row_number INTEGER NOT NULL,
        payload TEXT NOT NULL, PRIMARY KEY(source_path,table_name,row_number))""",
]


def migrated_namespace(catalog_path, db_path=None):
    """Read the explicit migration provenance without changing either database."""
    destination = account_path(db_path)
    if not destination.is_file():
        return None
    conn = connect(destination, readonly=True)
    try:
        if not table_exists(conn, "catalog_provenance"):
            return None
        row = conn.execute(
            "SELECT namespace FROM catalog_provenance WHERE catalog_path=?", (str(Path(catalog_path).resolve()),)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def create_schema(conn):
    """Caller owns the transaction; do not use executescript (it commits implicitly)."""
    if table_exists(conn, "accounts") and not table_exists(conn, "account_storage_meta"):
        raise MigrationRequired("Legacy accounts require the offline migration CLI")
    for statement in _STATEMENTS:
        conn.execute(statement)
    row = conn.execute("SELECT version FROM account_storage_meta").fetchone()
    if row is None:
        conn.execute("INSERT INTO account_storage_meta VALUES (?)", (SCHEMA_VERSION,))
    elif row[0] != SCHEMA_VERSION:
        raise MigrationRequired("Unsupported account storage version")


def check_pending_migration(destination):
    completed = set()
    if destination.is_file():
        conn = connect(destination, readonly=True)
        try:
            if table_exists(conn, "migration_sources"):
                completed = {r[0] for r in conn.execute("SELECT source_path FROM migration_sources")}
        finally:
            conn.close()
    for path in discover_legacy_sources():
        if str(path) in completed:
            continue
        conn = connect(path, readonly=True)
        try:
            if table_exists(conn, "accounts") and conn.execute("SELECT 1 FROM accounts LIMIT 1").fetchone():
                raise MigrationRequired(
                    "Legacy account data found. Stop both apps and run "
                    "python -m src.account_migration plan, then apply with backups."
                )
        finally:
            conn.close()


def get_connection(db_path=None):
    path = account_path(db_path)
    if db_path is None:
        check_pending_migration(path)
    conn = connect(path)
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            create_schema(conn)
        return conn
    except Exception:
        conn.close()
        raise


def require_active(conn, account_id):
    if conn.execute("SELECT 1 FROM accounts WHERE account_id=? AND is_active=1", (int(account_id),)).fetchone() is None:
        raise ValueError("An active account is required")


def log_account_action(operation, administrator, target, *, conn):
    conn.execute(
        "INSERT INTO account_audit_log (ts,operation,administrator,target) VALUES (?,?,?,?)",
        (datetime.now(UTC).isoformat(timespec="seconds"), operation, administrator, target),
    )
