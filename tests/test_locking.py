"""Tests for optimistic locking on memory operations."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest

from memra_local.exceptions import ConcurrentModificationError
from memra_local.services.memory_service import MemoryService
from memra_local.storage.flat_file import FlatFileStore
from memra_local.storage.sqlite_index import SQLiteIndex


@pytest.fixture
def svc():
    """Create a MemoryService backed by a temporary storage directory."""
    tmp = Path(tempfile.mkdtemp(prefix="memra_locking_test_"))
    store = FlatFileStore(tmp)
    index = SQLiteIndex(tmp / "index.db")
    index.initialize()
    service = MemoryService(store, index)
    yield service
    index.close()
    shutil.rmtree(tmp, ignore_errors=True)


def _add_memory(svc: MemoryService, content: str, **kwargs) -> dict:
    """Helper to add a memory and return its data."""
    req = {"content": content, "tenant_id": "local", "project_id": "default", **kwargs}
    data, _ = svc.add(req)
    return data


class TestOptimisticLocking:
    def test_update_by_id_locked_success(self, svc: MemoryService):
        """SQLiteIndex.update_by_id_locked succeeds with correct revision."""
        mem = _add_memory(svc, "test locking")
        rc = svc.index.update_by_id_locked(mem["id"], {"importance": 8}, expected_revision=1)
        assert rc == 1
        row = svc.index.get_by_id(mem["id"])
        assert row["revision"] == 2

    def test_update_by_id_locked_conflict(self, svc: MemoryService):
        """SQLiteIndex.update_by_id_locked returns 0 on revision mismatch."""
        mem = _add_memory(svc, "test conflict")
        rc = svc.index.update_by_id_locked(mem["id"], {"importance": 8}, expected_revision=42)
        assert rc == 0

    def test_concurrent_modification_on_supersede(self, svc: MemoryService):
        """Simulating a revision mismatch on supersede raises ConcurrentModificationError."""
        mem = _add_memory(svc, "original version")
        # Manually bump revision to simulate concurrent modification
        svc.index.update_by_id_locked(mem["id"], {"importance": 6}, expected_revision=1)
        # Now revision is 2, but supersede will read revision 1 from get_by_id...
        # We need to test at MemoryService level -- this will be tested in Task 2
        # For now, test the index-level locking directly
        new = _add_memory(svc, "replacement")
        rc = svc.index.supersede_by_id(mem["id"], new["id"], expected_revision=1)
        assert rc == 0  # conflict: revision is now 2, not 1


# ---------------------------------------------------------------------------
# MemoryService-level optimistic locking tests
# ---------------------------------------------------------------------------

class TestMemoryServiceLocking:
    def test_concurrent_modification_raises(self, svc: MemoryService):
        """Superseding after concurrent modification raises ConcurrentModificationError."""
        from unittest.mock import patch

        mem = _add_memory(svc, "concurrent test")

        # Patch get_by_id to return stale revision after the first call
        original_get = svc.index.get_by_id
        call_count = [0]

        def patched_get(mid):
            row = original_get(mid)
            if row and mid == mem["id"]:
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call: return stale revision 1
                    # Meanwhile, bump revision to simulate concurrent write
                    svc.index.update_by_id_locked(mid, {"importance": 7}, expected_revision=1)
            return row

        with patch.object(svc.index, "get_by_id", side_effect=patched_get):
            with pytest.raises(ConcurrentModificationError):
                svc.supersede(mem["id"], "new content")

    def test_concurrent_supersede(self, svc: MemoryService):
        """Two concurrent supersedes on same memory -- second one raises."""
        mem = _add_memory(svc, "race condition test")
        svc.supersede(mem["id"], "first supersede wins")

        # Second supersede should fail because memory is already superseded
        with pytest.raises(ValueError, match="already superseded"):
            svc.supersede(mem["id"], "second supersede loses")

    def test_update_with_expected_revision_success(self, svc: MemoryService):
        """svc.update with expected_revision succeeds when revision matches."""
        mem = _add_memory(svc, "revision update test")
        result = svc.update(mem["id"], {"importance": 8}, expected_revision=1)
        assert result is not None
        assert result["importance"] == 8

    def test_update_with_expected_revision_conflict(self, svc: MemoryService):
        """svc.update with wrong expected_revision raises ConcurrentModificationError."""
        mem = _add_memory(svc, "revision conflict test")
        with pytest.raises(ConcurrentModificationError):
            svc.update(mem["id"], {"importance": 8}, expected_revision=999)


class TestUpdateIndexFirst:
    """update() must apply the index update (dedup/revision checks) BEFORE
    writing the flat file — a rejected update must leave the file untouched,
    otherwise file and index drift apart."""

    def test_version_conflict_leaves_flat_file_untouched(self, svc: MemoryService):
        mem = _add_memory(svc, "original content")
        row = svc.index.get_by_id(mem["id"])

        with pytest.raises(ConcurrentModificationError):
            svc.update(mem["id"], {"content": "drifted content"}, expected_revision=999)

        on_disk = svc.store.read(row["storage_path"])
        assert on_disk["content"] == "original content"

    def test_dedup_conflict_leaves_flat_file_untouched(self, svc: MemoryService):
        """Updating B's content to duplicate A's violates the dedup unique
        index — the flat file for B must keep its original content."""
        _add_memory(svc, "content a")
        mem_b = _add_memory(svc, "content b")
        row_b = svc.index.get_by_id(mem_b["id"])

        with pytest.raises(sqlite3.IntegrityError):
            svc.update(mem_b["id"], {"content": "content a"})

        on_disk = svc.store.read(row_b["storage_path"])
        assert on_disk["content"] == "content b"


class TestThreadSafety:
    def test_concurrent_adds_keep_index_and_fts_consistent(self, svc: MemoryService):
        """4 threads x 25 adds on the shared connection: no errors, no FTS drift.

        Regression test for the unlocked shared SQLite connection: concurrent
        writes interleaved transactions ("cannot start a transaction within a
        transaction") and left memories_index and memories_fts permanently out
        of sync, making memories invisible to text search.
        """
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            for i in range(25):
                try:
                    svc.add(
                        {
                            "content": f"concurrent memory worker{worker_id} item{i}",
                            "tenant_id": "local",
                            "project_id": "default",
                        }
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

        with svc.index._lock:
            index_count = svc.index._c.execute(
                "SELECT COUNT(*) FROM memories_index"
            ).fetchone()[0]
            fts_count = svc.index._c.execute(
                "SELECT COUNT(*) FROM memories_fts"
            ).fetchone()[0]
        assert index_count == 100
        assert fts_count == index_count
