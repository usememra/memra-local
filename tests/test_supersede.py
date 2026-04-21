"""Tests for supersession chains, filtering, and chain walking."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from memra_local.services.memory_service import MemoryService
from memra_local.storage.flat_file import FlatFileStore
from memra_local.storage.sqlite_index import SQLiteIndex


@pytest.fixture
def svc():
    """Create a MemoryService backed by a temporary storage directory."""
    tmp = Path(tempfile.mkdtemp(prefix="memra_supersede_test_"))
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


# ---------------------------------------------------------------------------
# SQLiteIndex-level supersession tests
# ---------------------------------------------------------------------------

class TestSQLiteIndexSupersede:
    def test_supersede_by_id(self, svc: MemoryService):
        """supersede_by_id sets status, superseded_by, increments revision."""
        mem = _add_memory(svc, "original fact")
        old_id = mem["id"]
        new = _add_memory(svc, "new fact")
        new_id = new["id"]

        rowcount = svc.index.supersede_by_id(old_id, new_id, expected_revision=1)
        assert rowcount == 1

        row = svc.index.get_by_id(old_id)
        assert row["status"] == "superseded"
        assert row["superseded_by"] == new_id
        assert row["revision"] == 2

    def test_supersede_by_id_wrong_revision(self, svc: MemoryService):
        """supersede_by_id with wrong revision returns 0."""
        mem = _add_memory(svc, "original")
        new = _add_memory(svc, "replacement")
        rowcount = svc.index.supersede_by_id(mem["id"], new["id"], expected_revision=999)
        assert rowcount == 0

    def test_get_chain_rows(self, svc: MemoryService):
        """get_chain_rows returns ordered list from root to current."""
        m1 = _add_memory(svc, "version 1")
        m2 = _add_memory(svc, "version 2")
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)

        chain = svc.index.get_chain_rows(m2["id"])
        assert len(chain) == 2
        assert chain[0]["id"] == m1["id"]
        assert chain[1]["id"] == m2["id"]

    def test_get_chain_rows_from_middle(self, svc: MemoryService):
        """Starting from middle of 3-item chain returns full chain."""
        m1 = _add_memory(svc, "v1")
        m2 = _add_memory(svc, "v2")
        m3 = _add_memory(svc, "v3")
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)
        svc.index.supersede_by_id(m2["id"], m3["id"], expected_revision=1)

        chain = svc.index.get_chain_rows(m2["id"])
        assert len(chain) == 3
        assert chain[0]["id"] == m1["id"]
        assert chain[1]["id"] == m2["id"]
        assert chain[2]["id"] == m3["id"]


# ---------------------------------------------------------------------------
# Filtering tests (SQLiteIndex level)
# ---------------------------------------------------------------------------

class TestSupersedeFiltering:
    def test_search_fts_excludes_superseded(self, svc: MemoryService):
        """search_fts returns only active memories, not superseded."""
        m1 = _add_memory(svc, "important python fact")
        m2 = _add_memory(svc, "updated python fact")
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)

        results = svc.index.search_fts("python", "default", "local")
        ids = [r["id"] for r in results]
        assert m1["id"] not in ids
        assert m2["id"] in ids

    def test_list_memories_excludes_superseded(self, svc: MemoryService):
        """list_memories excludes superseded by default."""
        m1 = _add_memory(svc, "fact one")
        m2 = _add_memory(svc, "fact two")
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)

        rows, total = svc.index.list_memories(namespace="default", tenant_id="local")
        ids = [r["id"] for r in rows]
        assert m1["id"] not in ids
        assert m2["id"] in ids
        assert total == 1

    def test_list_memories_includes_superseded(self, svc: MemoryService):
        """list_memories with include_superseded=True returns all."""
        m1 = _add_memory(svc, "fact alpha")
        m2 = _add_memory(svc, "fact beta")
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)

        rows, total = svc.index.list_memories(
            namespace="default", tenant_id="local", include_superseded=True
        )
        ids = [r["id"] for r in rows]
        assert m1["id"] in ids
        assert m2["id"] in ids
        assert total == 2

    def test_get_candidates_excludes_superseded(self, svc: MemoryService):
        """get_candidates_with_embeddings excludes superseded."""
        m1 = _add_memory(svc, "candidate one")
        m2 = _add_memory(svc, "candidate two")
        # Add fake embeddings
        svc.index.update_embedding(m1["id"], b"\x00" * 16)
        svc.index.update_embedding(m2["id"], b"\x00" * 16)
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)

        candidates = svc.index.get_candidates_with_embeddings("default", "local")
        ids = [c["id"] for c in candidates]
        assert m1["id"] not in ids
        assert m2["id"] in ids

    def test_find_by_hash_excludes_superseded(self, svc: MemoryService):
        """find_by_hash returns None for superseded memory."""
        import hashlib
        m1 = _add_memory(svc, "unique content xyz")
        m2 = _add_memory(svc, "replacement content")
        svc.index.supersede_by_id(m1["id"], m2["id"], expected_revision=1)

        content_hash = hashlib.sha256("unique content xyz".encode()).hexdigest()
        result = svc.index.find_by_hash("default", "local", content_hash)
        assert result is None


# ---------------------------------------------------------------------------
# Optimistic locking on update (SQLiteIndex level)
# ---------------------------------------------------------------------------

class TestOptimisticLockIndex:
    def test_optimistic_lock_update_success(self, svc: MemoryService):
        """update_by_id_locked succeeds when revision matches."""
        mem = _add_memory(svc, "lockable content")
        rowcount = svc.index.update_by_id_locked(
            mem["id"], {"importance": 9}, expected_revision=1
        )
        assert rowcount == 1
        row = svc.index.get_by_id(mem["id"])
        assert row["importance"] == 9
        assert row["revision"] == 2

    def test_optimistic_lock_update_failure(self, svc: MemoryService):
        """update_by_id_locked fails when revision mismatched."""
        mem = _add_memory(svc, "lockable content too")
        rowcount = svc.index.update_by_id_locked(
            mem["id"], {"importance": 9}, expected_revision=999
        )
        assert rowcount == 0
        row = svc.index.get_by_id(mem["id"])
        assert row["importance"] == 5  # unchanged


# ---------------------------------------------------------------------------
# MemoryService-level supersession tests
# ---------------------------------------------------------------------------

class TestMemoryServiceSupersede:
    def test_supersede_creates_new_memory(self, svc: MemoryService):
        """svc.supersede() returns (new_data, old_data) with correct state."""
        old = _add_memory(svc, "original content")
        new_data, old_data = svc.supersede(old["id"], "updated content")

        assert new_data["content"] == "updated content"
        assert new_data["id"] != old["id"]
        assert old_data["status"] == "superseded"
        assert old_data["superseded_by"] == new_data["id"]

    def test_supersede_nonexistent_raises(self, svc: MemoryService):
        """Superseding a nonexistent memory raises ValueError."""
        with pytest.raises(ValueError, match="Memory not found"):
            svc.supersede("nonexistent_id", "content")

    def test_supersede_already_superseded_raises(self, svc: MemoryService):
        """Superseding an already-superseded memory raises ValueError."""
        m1 = _add_memory(svc, "first version")
        svc.supersede(m1["id"], "second version")

        with pytest.raises(ValueError, match="already superseded"):
            svc.supersede(m1["id"], "third version")

    def test_recall_excludes_superseded(self, svc: MemoryService):
        """After superseding, recall returns only the new memory."""
        m1 = _add_memory(svc, "recall test original")
        new_data, _ = svc.supersede(m1["id"], "recall test updated")

        results = svc.search("recall test", "default", "local")
        ids = [r["id"] for r in results]
        assert m1["id"] not in ids
        assert new_data["id"] in ids

    def test_bootstrap_excludes_superseded(self, svc: MemoryService):
        """Bootstrap returns only non-superseded memories."""
        m1 = _add_memory(svc, "bootstrap test original", importance=9)
        new_data, _ = svc.supersede(m1["id"], "bootstrap test updated")

        results = svc.bootstrap("default", "local")
        ids = [r["id"] for r in results]
        assert m1["id"] not in ids
        assert new_data["id"] in ids

    def test_list_excludes_superseded(self, svc: MemoryService):
        """list_ returns only active memories."""
        m1 = _add_memory(svc, "list test original")
        new_data, _ = svc.supersede(m1["id"], "list test updated")

        items, total = svc.list_(namespace="default", tenant_id="local")
        ids = [i["id"] for i in items]
        assert m1["id"] not in ids
        assert new_data["id"] in ids

    def test_chain_walking(self, svc: MemoryService):
        """get_chain returns ordered list oldest to newest."""
        m1 = _add_memory(svc, "chain v1")
        new_data, _ = svc.supersede(m1["id"], "chain v2")

        chain = svc.get_chain(new_data["id"])
        assert len(chain) == 2
        assert chain[0]["content"] == "chain v1"
        assert chain[1]["content"] == "chain v2"

    def test_chain_from_middle(self, svc: MemoryService):
        """Calling get_chain from middle of 3-item chain returns full chain."""
        m1 = _add_memory(svc, "chain3 v1")
        m2_data, _ = svc.supersede(m1["id"], "chain3 v2")
        m3_data, _ = svc.supersede(m2_data["id"], "chain3 v3")

        chain = svc.get_chain(m2_data["id"])
        assert len(chain) == 3
        assert chain[0]["content"] == "chain3 v1"
        assert chain[1]["content"] == "chain3 v2"
        assert chain[2]["content"] == "chain3 v3"
