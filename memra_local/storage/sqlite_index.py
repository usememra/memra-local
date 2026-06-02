"""SQLite WAL index with FTS5 text search for memory metadata."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _fts5_phrase(query: str) -> str:
    """Escape a user query for safe use as an FTS5 MATCH phrase.

    SQLite parameter binding escapes SQL injection but NOT FTS5 query
    syntax — operators like ``-``, ``:``, ``*``, ``OR``, ``AND``, ``NEAR``
    are interpreted by the FTS5 parser. Wrapping the query in phrase
    quotes neutralizes them and treats the whole string as a literal
    phrase. Internal ``"`` is escaped by doubling (FTS5 convention).
    """
    if not query or not query.strip():
        return '""'
    return '"' + query.replace('"', '""') + '"'


class SQLiteIndex:
    """SQLite-backed memory index with WAL journaling and FTS5 full-text search."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create database, enable WAL mode, create tables and indexes."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories_index (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL DEFAULT 'default',
                tenant_id TEXT NOT NULL DEFAULT 'local',
                type TEXT NOT NULL DEFAULT 'fact',
                importance INTEGER NOT NULL DEFAULT 5,
                tags TEXT NOT NULL DEFAULT '[]',
                content_hash TEXT NOT NULL,
                source TEXT,
                metadata TEXT,
                storage_path TEXT NOT NULL,
                expires_at TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_namespace
                ON memories_index(namespace);

            CREATE INDEX IF NOT EXISTS idx_memories_tenant
                ON memories_index(namespace, tenant_id);

            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories_index(type);

            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories_index(importance);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_dedup
                ON memories_index(namespace, tenant_id, content_hash);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(id UNINDEXED, content, tokenize='porter unicode61');

            CREATE TABLE IF NOT EXISTS sync_events (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                event_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                base_revision INTEGER,
                pushed_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sync_events_namespace
                ON sync_events(namespace);

            CREATE INDEX IF NOT EXISTS idx_sync_events_unpushed
                ON sync_events(namespace, pushed_at)
                WHERE pushed_at IS NULL;

            CREATE TABLE IF NOT EXISTS sync_cursors (
                namespace TEXT PRIMARY KEY,
                cloud_api_url TEXT NOT NULL DEFAULT 'https://usememra.com/api/v1',
                remote_cursor INTEGER NOT NULL DEFAULT 0,
                last_synced_at TEXT,
                enabled_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

        # Migration: add embedding BLOB column if not present
        cursor = self._conn.execute("PRAGMA table_info(memories_index)")
        columns = {row[1] for row in cursor.fetchall()}
        if "embedding" not in columns:
            self._conn.execute(
                "ALTER TABLE memories_index ADD COLUMN embedding BLOB"
            )
            self._conn.commit()

        if "revision" not in columns:
            self._conn.execute(
                "ALTER TABLE memories_index ADD COLUMN revision INTEGER DEFAULT 1"
            )
            self._conn.commit()

        if "superseded_by" not in columns:
            self._conn.execute(
                "ALTER TABLE memories_index ADD COLUMN superseded_by TEXT"
            )
            self._conn.commit()

        # Index on superseded_by — required for get_chain_rows recursive CTE
        # to scale linearly (without it the ancestor join falls back to a
        # full scan per recursion step → O(N²) on deep chains).
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_superseded_by "
            "ON memories_index(superseded_by)"
        )
        self._conn.commit()

        # Migration: add pii_mode column to sync_cursors if not present
        cursor_cols = self._conn.execute("PRAGMA table_info(sync_cursors)")
        sync_cursor_columns = {row[1] for row in cursor_cols.fetchall()}
        if "pii_mode" not in sync_cursor_columns:
            self._conn.execute(
                "ALTER TABLE sync_cursors ADD COLUMN pii_mode TEXT NOT NULL DEFAULT 'shared_masked'"
            )
            self._conn.commit()

        # Migration (77-04): add agent_id column so cloud_bootstrap knows
        # which /v1/agents/{agent_id}/bootstrap to call. Nullable — if unset,
        # cloud bootstrap is skipped (local-only behaviour preserved).
        if "agent_id" not in sync_cursor_columns:
            self._conn.execute(
                "ALTER TABLE sync_cursors ADD COLUMN agent_id TEXT"
            )
            self._conn.commit()

    @property
    def _c(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteIndex not initialized. Call initialize() first.")
        return self._conn

    def insert(
        self,
        memory_id: str,
        namespace: str,
        tenant_id: str,
        type_: str,
        importance: int,
        tags: list[str],
        content_hash: str,
        storage_path: str,
        content: str,
        source: str | None,
        metadata: dict | None,
        created_at: str,
        updated_at: str,
    ) -> None:
        """Insert a memory into the index and FTS table.

        Raises:
            sqlite3.IntegrityError: If content_hash already exists for namespace+tenant_id
        """
        self._c.execute(
            """INSERT INTO memories_index
               (id, namespace, tenant_id, type, importance, tags, content_hash,
                source, metadata, storage_path, expires_at, confidence, status,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1.0, 'active', ?, ?)""",
            (
                memory_id,
                namespace,
                tenant_id,
                type_,
                importance,
                json.dumps(tags),
                content_hash,
                source,
                json.dumps(metadata) if metadata else None,
                storage_path,
                created_at,
                updated_at,
            ),
        )
        self._c.execute(
            "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
            (memory_id, content),
        )
        self._c.commit()

    def find_by_hash(
        self, namespace: str, tenant_id: str, content_hash: str
    ) -> dict | None:
        """Find a memory by its content hash within a namespace+tenant scope."""
        row = self._c.execute(
            """SELECT * FROM memories_index
               WHERE namespace = ? AND tenant_id = ? AND content_hash = ?
               AND status != 'superseded'""",
            (namespace, tenant_id, content_hash),
        ).fetchone()
        return dict(row) if row else None

    def find_by_path(self, storage_path: str) -> dict | None:
        """Find an index entry by its storage_path."""
        row = self._c.execute(
            "SELECT * FROM memories_index WHERE storage_path = ?", (storage_path,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_by_id(self, memory_id: str) -> dict | None:
        """Get a memory by its ID."""
        row = self._c.execute(
            "SELECT * FROM memories_index WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_fts(
        self,
        query: str,
        namespace: str,
        tenant_id: str,
        type_: str | None = None,
        importance_min: int | None = None,
        limit: int = 20,
        offset: int = 0,
        include_proposed: bool = False,
    ) -> list[dict]:
        """Search memories using FTS5 full-text search.

        Args:
            query: Search query string
            namespace: Filter by namespace
            tenant_id: Filter by tenant
            type_: Optional type filter
            importance_min: Optional minimum importance filter
            limit: Max results to return
            offset: Number of results to skip
            include_proposed: If True, include trust_state='proposed' memories

        Returns:
            List of matching memory rows as dicts
        """
        conditions = ["m.namespace = ?", "m.tenant_id = ?", "m.status != 'superseded'"]
        params: list = [namespace, tenant_id]

        if type_ is not None:
            conditions.append("m.type = ?")
            params.append(type_)

        if importance_min is not None:
            conditions.append("m.importance >= ?")
            params.append(importance_min)

        # Trust state filtering
        if include_proposed:
            conditions.append(
                "(m.metadata IS NULL"
                " OR json_extract(m.metadata, '$.trust_state') IS NULL"
                " OR json_extract(m.metadata, '$.trust_state') IN ('verified', 'proposed'))"
            )
        else:
            conditions.append(
                "(m.metadata IS NULL"
                " OR json_extract(m.metadata, '$.trust_state') IS NULL"
                " OR json_extract(m.metadata, '$.trust_state') = 'verified')"
            )

        where_clause = " AND ".join(conditions)
        params.extend([_fts5_phrase(query), limit, offset])

        sql = f"""
            SELECT m.*, fts.rank
            FROM memories_fts fts
            JOIN memories_index m ON m.id = fts.id
            WHERE {where_clause}
              AND fts.content MATCH ?
            ORDER BY fts.rank
            LIMIT ? OFFSET ?
        """

        rows = self._c.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def list_memories(
        self,
        namespace: str | None = None,
        tenant_id: str | None = None,
        type_: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_superseded: bool = False,
    ) -> tuple[list[dict], int]:
        """List memories with optional filters.

        Returns:
            Tuple of (rows, total_count)
        """
        conditions: list[str] = []
        params: list = []

        if not include_superseded:
            conditions.append("status != 'superseded'")

        if namespace is not None:
            conditions.append("namespace = ?")
            params.append(namespace)
        if tenant_id is not None:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if type_ is not None:
            conditions.append("type = ?")
            params.append(type_)

        where = "WHERE " + " AND ".join(conditions)

        # Count total
        count_row = self._c.execute(
            f"SELECT COUNT(*) FROM memories_index {where}", params
        ).fetchone()
        total = count_row[0]

        # Fetch page
        rows = self._c.execute(
            f"""SELECT * FROM memories_index {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        return [dict(r) for r in rows], total

    def delete_by_id(self, memory_id: str) -> bool:
        """Delete a memory from both index and FTS tables.

        Returns:
            True if a row was deleted, False if not found
        """
        cursor = self._c.execute(
            "DELETE FROM memories_index WHERE id = ?", (memory_id,)
        )
        self._c.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        self._c.commit()
        return cursor.rowcount > 0

    def update_by_id(self, memory_id: str, updates: dict) -> dict | None:
        """Update a memory's index row.

        Args:
            memory_id: The memory to update
            updates: Dict of column->value to update

        Returns:
            Updated row as dict, or None if not found
        """
        if not updates:
            return self.get_by_id(memory_id)

        set_clauses = []
        params = []
        for col, val in updates.items():
            set_clauses.append(f"{col} = ?")
            if isinstance(val, (list, dict)):
                params.append(json.dumps(val))
            else:
                params.append(val)
        params.append(memory_id)

        self._c.execute(
            f"UPDATE memories_index SET {', '.join(set_clauses)} WHERE id = ?",
            params,
        )
        self._c.commit()
        return self.get_by_id(memory_id)

    def insert_with_embedding(
        self,
        memory_id: str,
        namespace: str,
        tenant_id: str,
        type_: str,
        importance: int,
        tags: list[str],
        content_hash: str,
        storage_path: str,
        content: str,
        source: str | None,
        metadata: dict | None,
        created_at: str,
        updated_at: str,
        embedding: bytes | None = None,
    ) -> None:
        """Insert a memory into the index with optional embedding BLOB.

        Raises:
            sqlite3.IntegrityError: If content_hash already exists for namespace+tenant_id
        """
        self._c.execute(
            """INSERT INTO memories_index
               (id, namespace, tenant_id, type, importance, tags, content_hash,
                source, metadata, storage_path, expires_at, confidence, status,
                created_at, updated_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 1.0, 'active', ?, ?, ?)""",
            (
                memory_id,
                namespace,
                tenant_id,
                type_,
                importance,
                json.dumps(tags),
                content_hash,
                source,
                json.dumps(metadata) if metadata else None,
                storage_path,
                created_at,
                updated_at,
                embedding,
            ),
        )
        self._c.execute(
            "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
            (memory_id, content),
        )
        self._c.commit()

    def get_candidates_with_embeddings(
        self,
        namespace: str,
        tenant_id: str,
        type_: str | None = None,
        importance_min: int | None = None,
        limit: int = 200,
        include_proposed: bool = False,
    ) -> list[dict]:
        """Get candidate memories that have embeddings for cosine search.

        Args:
            namespace: Filter by namespace
            tenant_id: Filter by tenant
            type_: Optional type filter
            importance_min: Optional minimum importance filter
            limit: Max candidates to return
            include_proposed: If True, include trust_state='proposed' memories

        Returns:
            List of memory rows (as dicts) where embedding IS NOT NULL
        """
        conditions = ["namespace = ?", "tenant_id = ?", "embedding IS NOT NULL", "status != 'superseded'"]
        params: list = [namespace, tenant_id]

        if type_ is not None:
            conditions.append("type = ?")
            params.append(type_)

        if importance_min is not None:
            conditions.append("importance >= ?")
            params.append(importance_min)

        # Trust state filtering: exclude proposed/rejected by default.
        # Treat NULL metadata or missing trust_state as implicitly verified.
        if include_proposed:
            conditions.append(
                "(metadata IS NULL"
                " OR json_extract(metadata, '$.trust_state') IS NULL"
                " OR json_extract(metadata, '$.trust_state') IN ('verified', 'proposed'))"
            )
        else:
            conditions.append(
                "(metadata IS NULL"
                " OR json_extract(metadata, '$.trust_state') IS NULL"
                " OR json_extract(metadata, '$.trust_state') = 'verified')"
            )

        where = " AND ".join(conditions)
        params.append(limit)

        rows = self._c.execute(
            f"SELECT * FROM memories_index WHERE {where} LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def update_embedding(self, memory_id: str, embedding: bytes) -> None:
        """Update the embedding BLOB for an existing memory (backfill support).

        Args:
            memory_id: The memory to update
            embedding: Serialized embedding bytes (wrapped in sqlite3.Binary)
        """
        self._c.execute(
            "UPDATE memories_index SET embedding = ? WHERE id = ?",
            (embedding, memory_id),
        )
        self._c.commit()

    def supersede_by_id(
        self, old_id: str, new_id: str, expected_revision: int
    ) -> int:
        """Mark a memory as superseded with optimistic locking.

        Args:
            old_id: The memory to supersede
            new_id: The replacement memory ID
            expected_revision: Expected current revision (optimistic lock)

        Returns:
            Number of rows updated (0 = version conflict, 1 = success)
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cursor = self._c.execute(
            """UPDATE memories_index
               SET superseded_by = ?, status = 'superseded',
                   revision = revision + 1, updated_at = ?
               WHERE id = ? AND revision = ?""",
            (new_id, now, old_id, expected_revision),
        )
        self._c.commit()
        return cursor.rowcount

    def get_chain_rows(self, memory_id: str) -> list[dict]:
        """Walk supersession chain and return ordered list from root to current.

        Args:
            memory_id: Any memory ID in the chain

        Returns:
            List of memory row dicts, ordered oldest (root) to newest
        """
        # Single recursive CTE: walk ancestors (backward via superseded_by
        # predecessors) and descendants (forward via superseded_by) from the
        # anchor row, then join to memories_index and order oldest→newest.
        # One query, no per-row get_by_id loop — required for SC-5 (10K chain
        # returns in <100ms).
        rows = self._c.execute(
            """
            WITH RECURSIVE
              ancestors(id, depth) AS (
                SELECT id, 0 FROM memories_index WHERE id = :mid
                UNION ALL
                SELECT m.id, a.depth + 1
                FROM memories_index m
                JOIN ancestors a ON m.superseded_by = a.id
                WHERE a.depth < 100000
              ),
              descendants(id, depth) AS (
                SELECT id, 0 FROM memories_index WHERE id = :mid
                UNION ALL
                SELECT m.superseded_by, d.depth + 1
                FROM memories_index m
                JOIN descendants d ON m.id = d.id
                WHERE m.superseded_by IS NOT NULL AND d.depth < 100000
              )
            SELECT mi.*, chain.pos AS _chain_pos
            FROM memories_index mi
            JOIN (
              SELECT id, -depth AS pos FROM ancestors WHERE depth > 0
              UNION
              SELECT id, depth AS pos FROM descendants
            ) chain ON chain.id = mi.id
            ORDER BY chain.pos
            """,
            {"mid": memory_id},
        ).fetchall()

        return [dict(r) for r in rows]

    def update_by_id_locked(
        self, memory_id: str, updates: dict, expected_revision: int
    ) -> int:
        """Update a memory's index row with optimistic locking.

        Args:
            memory_id: The memory to update
            updates: Dict of column->value to update
            expected_revision: Expected current revision

        Returns:
            Number of rows updated (0 = version conflict, 1 = success)
        """
        if not updates:
            return 1

        set_clauses = ["revision = revision + 1"]
        params = []
        for col, val in updates.items():
            set_clauses.append(f"{col} = ?")
            if isinstance(val, (list, dict)):
                params.append(json.dumps(val))
            else:
                params.append(val)
        params.extend([memory_id, expected_revision])

        cursor = self._c.execute(
            f"UPDATE memories_index SET {', '.join(set_clauses)} WHERE id = ? AND revision = ?",
            params,
        )
        self._c.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
