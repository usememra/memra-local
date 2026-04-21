"""Tests for flat-file store and SQLite index with dual-write."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from memra_local.storage.flat_file import FlatFileStore
from memra_local.storage.sqlite_index import SQLiteIndex


class TestFlatFileStore:
    def test_flat_file_write(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        data = {"id": "mem-001", "content": "hello world", "type": "fact"}
        path = store.write("default", "mem-001", data)
        assert path == "default/mem-001.yaml"
        assert (tmp_storage / "default" / "mem-001.yaml").exists()

    def test_flat_file_read(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        data = {"id": "mem-002", "content": "read test", "type": "fact", "importance": 7}
        store.write("default", "mem-002", data)
        result = store.read("default/mem-002.yaml")
        assert result["id"] == "mem-002"
        assert result["content"] == "read test"
        assert result["importance"] == 7

    def test_flat_file_atomic(self, tmp_storage: Path) -> None:
        """Written file appears atomically (no .tmp files left)."""
        store = FlatFileStore(base_dir=tmp_storage)
        data = {"id": "mem-003", "content": "atomic test"}
        store.write("ns", "mem-003", data)
        # No temp files should remain
        tmp_files = list((tmp_storage / "ns").glob("*.tmp"))
        assert len(tmp_files) == 0
        # Final file exists with correct content
        result = store.read("ns/mem-003.yaml")
        assert result["content"] == "atomic test"

    def test_flat_file_delete(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        data = {"id": "mem-004", "content": "delete me"}
        store.write("default", "mem-004", data)
        assert store.exists("default/mem-004.yaml")
        deleted = store.delete("default/mem-004.yaml")
        assert deleted is True
        assert not store.exists("default/mem-004.yaml")

    def test_flat_file_delete_nonexistent(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        deleted = store.delete("default/nonexistent.yaml")
        assert deleted is False

    def test_flat_file_read_nonexistent(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        with pytest.raises(FileNotFoundError):
            store.read("default/nonexistent.yaml")

    def test_flat_file_rejects_namespace_traversal(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        with pytest.raises(ValueError):
            store.write("../escape", "mem-005", {"id": "mem-005"})

    def test_flat_file_rejects_read_path_escape(self, tmp_storage: Path) -> None:
        store = FlatFileStore(base_dir=tmp_storage)
        with pytest.raises(ValueError):
            store.read("../escape.yaml")


class TestSQLiteIndex:
    def test_sqlite_init(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "memories_index" in tables
        assert "memories_fts" in tables
        conn.close()
        idx.close()

    def test_sqlite_wal_mode(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        mode = idx._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        idx.close()

    def test_sqlite_insert_and_query(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        now = "2026-04-08T00:00:00Z"
        idx.insert(
            memory_id="mem-001",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=["test"],
            content_hash="abc123",
            storage_path="default/mem-001.yaml",
            content="hello world",
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )
        rows, total = idx.list_memories(namespace="default", tenant_id="local")
        assert total == 1
        assert rows[0]["id"] == "mem-001"
        idx.close()

    def test_sqlite_fts5_search(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        now = "2026-04-08T00:00:00Z"
        idx.insert(
            memory_id="mem-010",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="hash010",
            storage_path="default/mem-010.yaml",
            content="The quick brown fox jumps over the lazy dog",
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )
        idx.insert(
            memory_id="mem-011",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="hash011",
            storage_path="default/mem-011.yaml",
            content="Python is a great programming language",
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )
        results = idx.search_fts("fox", namespace="default", tenant_id="local")
        assert len(results) == 1
        assert results[0]["id"] == "mem-010"
        idx.close()

    def test_sqlite_dedup(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        now = "2026-04-08T00:00:00Z"
        idx.insert(
            memory_id="mem-020",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="duplicate_hash",
            storage_path="default/mem-020.yaml",
            content="duplicate content",
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )
        # Second insert with same namespace+tenant+content_hash should raise
        with pytest.raises(sqlite3.IntegrityError):
            idx.insert(
                memory_id="mem-021",
                namespace="default",
                tenant_id="local",
                type_="fact",
                importance=5,
                tags=[],
                content_hash="duplicate_hash",
                storage_path="default/mem-021.yaml",
                content="duplicate content",
                source=None,
                metadata=None,
                created_at=now,
                updated_at=now,
            )
        idx.close()

    def test_sqlite_get_by_id(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        now = "2026-04-08T00:00:00Z"
        idx.insert(
            memory_id="mem-030",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=8,
            tags=["important"],
            content_hash="hash030",
            storage_path="default/mem-030.yaml",
            content="get by id test",
            source="test",
            metadata={"key": "value"},
            created_at=now,
            updated_at=now,
        )
        row = idx.get_by_id("mem-030")
        assert row is not None
        assert row["id"] == "mem-030"
        assert row["importance"] == 8
        assert row is not None
        assert json.loads(row["tags"]) == ["important"]
        idx.close()

    def test_sqlite_find_by_hash(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        now = "2026-04-08T00:00:00Z"
        idx.insert(
            memory_id="mem-040",
            namespace="ns1",
            tenant_id="t1",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="findhash",
            storage_path="ns1/mem-040.yaml",
            content="find by hash",
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )
        found = idx.find_by_hash("ns1", "t1", "findhash")
        assert found is not None
        assert found["id"] == "mem-040"
        not_found = idx.find_by_hash("ns1", "t1", "nope")
        assert not_found is None
        idx.close()

    def test_sqlite_delete_by_id(self, tmp_storage: Path) -> None:
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()
        now = "2026-04-08T00:00:00Z"
        idx.insert(
            memory_id="mem-050",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="hash050",
            storage_path="default/mem-050.yaml",
            content="delete test",
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )
        deleted = idx.delete_by_id("mem-050")
        assert deleted is True
        assert idx.get_by_id("mem-050") is None
        # FTS should also be cleaned
        results = idx.search_fts("delete", namespace="default", tenant_id="local")
        assert len(results) == 0
        idx.close()


class TestDualWrite:
    def test_dual_write(self, tmp_storage: Path) -> None:
        """Write to both flat file and index, verify both contain the data."""
        store = FlatFileStore(base_dir=tmp_storage)
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()

        content = "dual write test memory"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        now = "2026-04-08T00:00:00Z"
        memory_id = "mem-dual-001"
        namespace = "default"

        data = {
            "id": memory_id,
            "content": content,
            "tenant_id": "local",
            "project_id": "default",
            "type": "fact",
            "importance": 5,
            "tags": [],
            "created_at": now,
            "updated_at": now,
        }

        # Write flat file
        storage_path = store.write(namespace, memory_id, data)

        # Write index
        idx.insert(
            memory_id=memory_id,
            namespace=namespace,
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash=content_hash,
            storage_path=storage_path,
            content=content,
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )

        # Verify flat file
        file_data = store.read(storage_path)
        assert file_data["content"] == content

        # Verify index
        row = idx.get_by_id(memory_id)
        assert row is not None
        assert row["storage_path"] == storage_path

        idx.close()

    def test_dual_write_cleanup_on_index_failure(self, tmp_storage: Path) -> None:
        """If index insert fails after file write, file should be cleaned up."""
        store = FlatFileStore(base_dir=tmp_storage)
        db_path = tmp_storage / "index.db"
        idx = SQLiteIndex(db_path=db_path)
        idx.initialize()

        content = "cleanup test"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        now = "2026-04-08T00:00:00Z"
        namespace = "default"

        # First write succeeds
        data1 = {"id": "mem-dup-1", "content": content}
        path1 = store.write(namespace, "mem-dup-1", data1)
        idx.insert(
            memory_id="mem-dup-1",
            namespace=namespace,
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash=content_hash,
            storage_path=path1,
            content=content,
            source=None,
            metadata=None,
            created_at=now,
            updated_at=now,
        )

        # Second write: file succeeds but index fails (duplicate hash)
        data2 = {"id": "mem-dup-2", "content": content}
        path2 = store.write(namespace, "mem-dup-2", data2)
        assert store.exists(path2)

        with pytest.raises(sqlite3.IntegrityError):
            idx.insert(
                memory_id="mem-dup-2",
                namespace=namespace,
                tenant_id="local",
                type_="fact",
                importance=5,
                tags=[],
                content_hash=content_hash,
                storage_path=path2,
                content=content,
                source=None,
                metadata=None,
                created_at=now,
                updated_at=now,
            )

        # Clean up the orphaned file
        store.delete(path2)
        assert not store.exists(path2)

        idx.close()
