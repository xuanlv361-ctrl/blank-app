"""Durable movie references and a read-only projection onto one catalog."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import defaultdict
from pathlib import Path

from .account_storage import ROOT, connect, migrated_namespace, table_exists


class IdentityConflict(ValueError):
    """An alias needs explicit reconciliation; messages contain no row values."""


def positive_id(value):
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number > 0 and number.is_integer() else None
    except (TypeError, ValueError, OverflowError):
        return None


def validate_stable_key(key):
    if not isinstance(key, str) or not re.fullmatch(
        r"(?:tmdb:[1-9][0-9]*|source:[A-Za-z0-9_.-]+:[1-9][0-9]*|(?:local|legacy):[a-f0-9-]+(?::[1-9][0-9]*)?)", key
    ):
        raise ValueError("Invalid durable movie identifier")


def namespace_for(path):
    path = Path(path).resolve()
    if path == (ROOT / "data" / "processed" / "app.db").resolve():
        return "movielens-100k"
    manifest = path.parent / "manifest.json"
    if path.parent.parent.name == "movielens-32m-21c" and manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        namespace = payload.get("identity_namespace", "")
        if re.fullmatch(r"(?:movielens-32m|local-ml32m\.[a-f0-9]{64})", namespace):
            return namespace
    if path.parent.parent.name == "letterboxd-2025" and manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        digest = payload.get("source_sha256", "")
        if re.fullmatch(r"[a-f0-9]{64}", str(digest)):
            return "letterboxd-2025." + digest
    # Unknown provenance is isolated, never assumed to share integer IDs.
    return "isolated." + hashlib.sha256(str(path).encode()).hexdigest()


def initialize_catalog(conn, path, *, namespace=None):
    conn.execute("CREATE TABLE IF NOT EXISTS catalog_identity (namespace TEXT NOT NULL)")
    row = conn.execute("SELECT namespace FROM catalog_identity").fetchone()
    if row is None:
        namespace = namespace or namespace_for(path)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", namespace):
            raise ValueError("Invalid catalog namespace")
        conn.execute("INSERT INTO catalog_identity VALUES (?)", (namespace,))
    elif namespace is not None and row[0] != namespace:
        raise IdentityConflict("Catalog provenance cannot be changed in place")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS catalog_movie_ids (movie_id INTEGER PRIMARY KEY, source_key TEXT UNIQUE NOT NULL)"
    )
    namespace = conn.execute("SELECT namespace FROM catalog_identity").fetchone()[0]
    local_keys = legacy_local_keys(conn, path)
    for row in conn.execute("SELECT movie_id,data FROM movies"):
        data = json.loads(row["data"])
        mid = int(row["movie_id"])
        source_id = positive_id(data.get("source_movie_id")) or mid
        key = local_keys.get(mid, f"source:{namespace}:{source_id}")
        conn.execute("INSERT OR IGNORE INTO catalog_movie_ids VALUES (?,?)", (mid, key))
        if mid in local_keys:
            conn.execute(
                "UPDATE catalog_movie_ids SET source_key=? WHERE movie_id=? AND source_key LIKE 'source:%'", (key, mid)
            )


def legacy_local_keys(conn, path):
    """Legacy add_movie events identify local records, never dataset integer IDs."""
    keys = {}
    if table_exists(conn, "admin_audit_log"):
        for row in conn.execute("SELECT id,ts,target FROM admin_audit_log WHERE operation='add_movie' ORDER BY id"):
            match = re.fullmatch(r"movie_id=([1-9][0-9]*)", str(row["target"]))
            if match:
                mid = int(match[1])
                token = json.dumps([str(Path(path).resolve()), mid, row["id"], row["ts"]])
                keys[mid] = "local:" + hashlib.sha256(token.encode()).hexdigest()
    return keys


def add_local_identity(conn, movie_id):
    conn.execute("INSERT INTO catalog_movie_ids VALUES (?,?)", (int(movie_id), "local:" + uuid.uuid4().hex))


class CatalogIdentity:
    def __init__(self, records=(), *, namespace="isolated", source_keys=None):
        self.namespace = namespace
        self.rows = {int(row["movie_id"]): row for row in records}
        self.sources = {}
        self.primary = {}
        self.titles = {}
        self.conflicts = set()
        self.by_alias = defaultdict(set)
        tmdb_members = defaultdict(list)
        for mid, row in self.rows.items():
            source_id = positive_id(row.get("source_movie_id")) or mid
            source = (source_keys or {}).get(mid, row.get("catalog_source_key") or f"source:{namespace}:{source_id}")
            self.sources[mid] = source
            self.titles[mid] = str(row.get("title", ""))
            self.primary[mid] = source
            self.by_alias[source].add(mid)
            tmdb = positive_id(row.get("tmdb_id"))
            if (
                not source.startswith("local:")
                and tmdb
                and row.get("metadata_status") in {"Matched title and year", "GroupLens links.csv"}
            ):
                tmdb_members[f"tmdb:{tmdb}"].append(mid)
        for key, members in tmdb_members.items():
            if len(members) != 1:
                self.conflicts.add(key)
                continue
            mid = members[0]
            self.primary[mid] = key
            self.by_alias[key].add(mid)

    @classmethod
    def from_connection(cls, conn, path, *, namespace=None):
        if table_exists(conn, "catalog_identity"):
            row = conn.execute("SELECT namespace FROM catalog_identity").fetchone()
            if row:
                if namespace is not None and namespace != row[0]:
                    raise IdentityConflict("Catalog provenance conflicts with the requested namespace")
                namespace = row[0]
        namespace = namespace or namespace_for(path)
        source_keys = {}
        if table_exists(conn, "catalog_movie_ids"):
            source_keys = {int(r[0]): r[1] for r in conn.execute("SELECT movie_id,source_key FROM catalog_movie_ids")}
        for mid, key in legacy_local_keys(conn, path).items():
            if not source_keys.get(mid, "").startswith("local:"):
                source_keys[mid] = key
        rows = []
        if table_exists(conn, "movies"):
            for row in conn.execute("SELECT movie_id,data FROM movies ORDER BY movie_id"):
                data = json.loads(row["data"])
                data["movie_id"] = int(row["movie_id"])
                rows.append(data)
        return cls(rows, namespace=namespace, source_keys=source_keys)

    @classmethod
    def from_path(cls, path, *, namespace=None):
        if not Path(path).is_file():
            return cls()
        conn = connect(path, readonly=True)
        try:
            return cls.from_connection(conn, path, namespace=namespace)
        finally:
            conn.close()

    def reference(self, value):
        if isinstance(value, int):
            value = f"catalog:{value}"
        if isinstance(value, str) and re.fullmatch(r"catalog:[1-9][0-9]*", value):
            mid = int(value.split(":")[1])
            if mid not in self.rows:
                raise IdentityConflict("Movie is unavailable in the selected catalog")
            source, key = self.sources[mid], self.primary[mid]
            if len(self.by_alias[source]) != 1:
                raise IdentityConflict("Multiple catalog movies share a source identity")
            return key, list(dict.fromkeys([source, key])), self.titles[mid]
        validate_stable_key(value)
        if value in self.conflicts:
            raise IdentityConflict("Ambiguous TMDB mapping requires explicit reconciliation")
        members = self.by_alias.get(value, set())
        if len(members) > 1:
            raise IdentityConflict("Multiple catalog movies share an identity")
        if members:
            return self.reference(f"catalog:{next(iter(members))}")
        return value, [value], ""

    def project(self, key, aliases=()):
        keys = set(aliases) | {key}
        if keys & self.conflicts:
            raise IdentityConflict("Ambiguous TMDB mapping requires explicit reconciliation")
        members = set().union(*(self.by_alias.get(alias, set()) for alias in keys))
        if len(members) > 1:
            raise IdentityConflict("Stored aliases map to different catalog movies")
        if not members:
            return key
        mid = next(iter(members))
        known_tmdb = {alias for alias in keys if alias.startswith("tmdb:")}
        current = self.primary[mid]
        if known_tmdb and current.startswith("tmdb:") and current not in known_tmdb:
            raise IdentityConflict("Verified movie mapping changed; reconciliation required")
        return f"catalog:{mid}"


def catalog_for(db_path=None, catalog_db_path=None):
    if catalog_db_path is not None:
        path = catalog_db_path
    elif db_path is not None:
        path = db_path
    else:
        from .db import DB_PATH

        path = DB_PATH
    namespace = migrated_namespace(path, db_path)
    return CatalogIdentity.from_path(path, namespace=namespace)


def alias_map(conn):
    result = defaultdict(list)
    for row in conn.execute("SELECT alias_key,item_key FROM item_aliases"):
        result[row["item_key"]].append(row["alias_key"])
    return result


def bind_reference(conn, reference):
    key, aliases, title = reference
    for alias in aliases:
        validate_stable_key(alias)
    existing = set()
    for alias in aliases:
        row = conn.execute("SELECT item_key FROM item_aliases WHERE alias_key=?", (alias,)).fetchone()
        if row:
            existing.add(row[0])
    if len(existing) > 1:
        raise IdentityConflict("Stored movie aliases conflict; explicit reconciliation required")
    if existing:
        key = next(iter(existing))
    known_tmdb = {
        r[0]
        for r in conn.execute("SELECT alias_key FROM item_aliases WHERE item_key=?", (key,))
        if r[0].startswith("tmdb:")
    }
    incoming_tmdb = {alias for alias in aliases if alias.startswith("tmdb:")}
    if known_tmdb and incoming_tmdb and known_tmdb != incoming_tmdb:
        raise IdentityConflict("Verified movie mapping changed; reconciliation required")
    conn.execute("INSERT OR IGNORE INTO item_refs VALUES (?,?)", (key, title))
    if title:
        conn.execute("UPDATE item_refs SET title=? WHERE item_key=? AND title=''", (title, key))
    for alias in aliases:
        conn.execute("INSERT OR IGNORE INTO item_aliases VALUES (?,?)", (alias, key))
    return key


def find_reference(conn, reference):
    key, aliases, _ = reference
    targets = {
        r[0] for alias in aliases for r in conn.execute("SELECT item_key FROM item_aliases WHERE alias_key=?", (alias,))
    }
    if len(targets) > 1:
        raise IdentityConflict("Stored movie aliases conflict; explicit reconciliation required")
    target = next(iter(targets), key)
    known = {
        r[0]
        for r in conn.execute("SELECT alias_key FROM item_aliases WHERE item_key=?", (target,))
        if r[0].startswith("tmdb:")
    }
    incoming = {alias for alias in aliases if alias.startswith("tmdb:")}
    if known and incoming and known != incoming:
        raise IdentityConflict("Verified movie mapping changed; reconciliation required")
    return target
