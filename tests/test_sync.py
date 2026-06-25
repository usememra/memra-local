"""Tests for sync service: enable, push, pull, conflicts, resolve."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from memra_local.services.factory import create_service
from memra_local.services.sync_service import SyncService


@pytest.fixture
def tmp_storage():
    d = Path(tempfile.mkdtemp(prefix="memra_sync_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def svc(tmp_storage):
    """Create a MemoryService with SyncService wired in."""
    return create_service(scope="global", storage_dir=tmp_storage)


@pytest.fixture
def sync(svc):
    """Get the SyncService from the MemoryService."""
    return svc.sync_service


# -------------------------------------------------------------------
# Enable / disable / status
# -------------------------------------------------------------------

class TestSyncEnable:
    def test_sync_enable_stores_config(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123")
        assert sync.is_sync_enabled("my-ns") is True

    def test_is_sync_enabled_false_for_unsynced(self, sync):
        assert sync.is_sync_enabled("not-synced") is False

    def test_disable_removes_config(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123")
        assert sync.disable("my-ns") is True
        assert sync.is_sync_enabled("my-ns") is False

    def test_get_status_returns_config(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1")
        status = sync.get_status("my-ns")
        assert status is not None
        assert status["namespace"] == "my-ns"
        assert status["cloud_api_url"] == "https://example.com/api/v1"
        assert status["remote_cursor"] == 0

    def test_get_status_returns_none_for_unsynced(self, sync):
        assert sync.get_status("not-synced") is None


# -------------------------------------------------------------------
# Record event
# -------------------------------------------------------------------

class TestRecordEvent:
    def test_record_event_inserts_for_synced_namespace(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123")
        sync.record_event("memory_created", "my-ns", "mem_123", {"content": "hello"})
        # Verify event exists
        row = sync._conn.execute(
            "SELECT * FROM sync_events WHERE namespace = ?", ("my-ns",)
        ).fetchone()
        assert row is not None
        assert row["event_type"] == "memory_created"
        assert row["memory_id"] == "mem_123"

    def test_record_event_noop_for_unsynced_namespace(self, sync):
        sync.record_event("memory_created", "not-synced", "mem_123", {"content": "hello"})
        row = sync._conn.execute(
            "SELECT COUNT(*) FROM sync_events WHERE namespace = ?", ("not-synced",)
        ).fetchone()
        assert row[0] == 0

    def test_record_event_never_crashes(self, sync):
        """Even with a broken connection, record_event should not raise."""
        sync.enable("my-ns", api_key="memra_live_test123")
        # Close connection to force error
        original_conn = sync._conn
        sync._conn = MagicMock()
        sync._conn.execute.side_effect = Exception("DB broken")
        # Should not raise
        sync.record_event("memory_created", "my-ns", "mem_123", {"content": "hello"})
        sync._conn = original_conn


# -------------------------------------------------------------------
# MemoryService hooks
# -------------------------------------------------------------------

class TestMemoryServiceHooks:
    def test_add_records_memory_created_event(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123")
        svc.add({"content": "test memory", "project_id": "test-ns"})
        row = sync._conn.execute(
            "SELECT * FROM sync_events WHERE namespace = ? AND event_type = ?",
            ("test-ns", "memory_created"),
        ).fetchone()
        assert row is not None

    def test_update_records_memory_updated_event(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123")
        data, _ = svc.add({"content": "original", "project_id": "test-ns"})
        svc.update(data["id"], {"content": "updated"})
        row = sync._conn.execute(
            "SELECT * FROM sync_events WHERE namespace = ? AND event_type = ?",
            ("test-ns", "memory_updated"),
        ).fetchone()
        assert row is not None

    def test_delete_records_memory_deleted_event(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123")
        data, _ = svc.add({"content": "to delete", "project_id": "test-ns"})
        svc.delete(data["id"])
        row = sync._conn.execute(
            "SELECT * FROM sync_events WHERE namespace = ? AND event_type = ?",
            ("test-ns", "memory_deleted"),
        ).fetchone()
        assert row is not None

    def test_unsynced_namespace_no_events(self, svc, sync):
        svc.add({"content": "not synced", "project_id": "unsynced-ns"})
        row = sync._conn.execute(
            "SELECT COUNT(*) FROM sync_events"
        ).fetchone()
        assert row[0] == 0


# -------------------------------------------------------------------
# Push
# -------------------------------------------------------------------

class TestPush:
    def test_push_sends_unpushed_events(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1", pii_mode="shared_raw")
        sync.record_event("memory_created", "my-ns", "mem_1", {"content": "hello"})
        sync.record_event("memory_created", "my-ns", "mem_2", {"content": "world"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cursor": 5, "pushed": 2, "conflicts": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(sync, "check_account_tier", return_value="team"):
            with patch("httpx.post", return_value=mock_response) as mock_post:
                result = sync.push("my-ns")

        assert result.pushed == 2
        assert result.conflicts == []
        assert result.cursor == 5
        mock_post.assert_called_once()

        # Verify events are marked pushed
        unpushed = sync._conn.execute(
            "SELECT COUNT(*) FROM sync_events WHERE namespace = ? AND pushed_at IS NULL",
            ("my-ns",),
        ).fetchone()
        assert unpushed[0] == 0

    def test_push_returns_error_on_network_failure(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", pii_mode="shared_raw")
        sync.record_event("memory_created", "my-ns", "mem_1", {"content": "hello"})

        with patch.object(sync, "check_account_tier", return_value="team"):
            with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
                result = sync.push("my-ns")

        assert result.error is not None
        assert result.pushed == 0

    def test_push_no_events_returns_zero(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123")
        result = sync.push("my-ns")
        assert result.pushed == 0
        assert result.error is None


# -------------------------------------------------------------------
# Pull
# -------------------------------------------------------------------

class TestPull:
    def test_pull_fetches_and_applies_events(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "events": [
                {
                    "event_id": "evt_1",
                    "event_type": "memory_created",
                    "memory_id": "mem_remote_1",
                    "payload": {
                        "content": "remote memory",
                        "type": "fact",
                        "importance": 5,
                        "project_id": "test-ns",
                    },
                },
            ],
            "cursor": 1,
            "has_more": False,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            result = sync.pull("test-ns", svc)

        assert result.applied == 1
        assert result.cursor == 1
        assert result.has_more is False

    def test_pull_returns_error_on_network_failure(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123")

        with patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            result = sync.pull("test-ns", svc)

        assert result.error is not None
        assert result.applied == 0

    @staticmethod
    def _mock_pull_response(events, cursor=1):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "events": events,
            "cursor": cursor,
            "has_more": False,
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response

    def test_pull_create_preserves_remote_id(self, svc, sync):
        """Applying a remote create must store the remote memory id locally."""
        sync.enable("test-ns", api_key="memra_live_test123")
        events = [
            {
                "event_id": "evt_1",
                "event_type": "memory_created",
                "memory_id": "mem_remote_xyz",
                "payload": {
                    "content": "remote memory",
                    "type": "fact",
                    "importance": 5,
                    "project_id": "test-ns",
                },
            },
        ]

        with patch("httpx.get", return_value=self._mock_pull_response(events)):
            result = sync.pull("test-ns", svc)

        assert result.applied == 1
        mem = svc.get("mem_remote_xyz")
        assert mem is not None
        assert mem["content"] == "remote memory"

    def test_pull_create_update_delete_roundtrip_same_row(self, svc, sync):
        """Remote create -> update -> delete must all land on the same local row.

        Regression: pull used to mint a fresh local ULID on create, so the
        follow-up update/delete events silently no-op'd while `applied` still
        incremented.
        """
        sync.enable("test-ns", api_key="memra_live_test123")
        events = [
            {
                "event_id": "evt_1",
                "event_type": "memory_created",
                "memory_id": "mem_remote_abc",
                "payload": {
                    "content": "remote v1",
                    "type": "fact",
                    "importance": 5,
                    "project_id": "test-ns",
                },
            },
            {
                "event_id": "evt_2",
                "event_type": "memory_updated",
                "memory_id": "mem_remote_abc",
                "payload": {"content": "remote v2"},
            },
            {
                "event_id": "evt_3",
                "event_type": "memory_deleted",
                "memory_id": "mem_remote_abc",
                "payload": {},
            },
        ]

        with patch("httpx.get", return_value=self._mock_pull_response(events, cursor=3)):
            result = sync.pull("test-ns", svc)

        assert result.applied == 3
        assert result.skipped == 0
        # Deleted at the end of the sequence
        assert svc.get("mem_remote_abc") is None

    def test_pull_update_delete_without_target_count_skipped(self, svc, sync):
        """update/delete events whose target is missing must count as skipped,
        not applied."""
        sync.enable("test-ns", api_key="memra_live_test123")
        events = [
            {
                "event_id": "evt_1",
                "event_type": "memory_updated",
                "memory_id": "mem_missing",
                "payload": {"content": "ghost"},
            },
            {
                "event_id": "evt_2",
                "event_type": "memory_deleted",
                "memory_id": "mem_missing",
                "payload": {},
            },
        ]

        with patch("httpx.get", return_value=self._mock_pull_response(events)):
            result = sync.pull("test-ns", svc)

        assert result.applied == 0
        assert result.skipped == 2


# -------------------------------------------------------------------
# Conflicts
# -------------------------------------------------------------------

class TestConflicts:
    def test_list_conflicts_returns_conflict_memories(self, svc, sync):
        # Simulate a conflict by creating a memory with _conflict_ in ID
        sync.enable("test-ns", api_key="memra_live_test123")
        data, _ = svc.add({"content": "original", "project_id": "test-ns"})
        original_id = data["id"]

        # Manually insert a conflict sibling in the index
        conflict_id = f"{original_id}_conflict_remote"
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        import hashlib
        conflict_content = "conflicting version"
        content_hash = hashlib.sha256(conflict_content.encode()).hexdigest()
        svc.store.write("test-ns", conflict_id, {
            "id": conflict_id, "content": conflict_content,
            "type": "fact", "importance": 5, "tags": [],
            "created_at": now, "updated_at": now,
            "tenant_id": "local", "project_id": "test-ns",
        })
        svc.index.insert_with_embedding(
            memory_id=conflict_id, namespace="test-ns", tenant_id="local",
            type_="fact", importance=5, tags=[], content_hash=content_hash,
            storage_path=f"test-ns/{conflict_id}.yaml", content=conflict_content,
            source=None, metadata=None, created_at=now, updated_at=now,
        )

        conflicts = sync.list_conflicts("test-ns")
        assert len(conflicts) >= 1
        assert any("_conflict_" in c["id"] for c in conflicts)

    def test_list_conflicts_empty_when_none(self, sync):
        conflicts = sync.list_conflicts("test-ns")
        assert conflicts == []

    def test_resolve_conflict_keep_local(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123")
        data, _ = svc.add({"content": "local version", "project_id": "test-ns"})
        original_id = data["id"]

        # Create conflict sibling
        conflict_id = f"{original_id}_conflict_remote"
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        import hashlib
        conflict_content = "remote version"
        content_hash = hashlib.sha256(conflict_content.encode()).hexdigest()
        svc.store.write("test-ns", conflict_id, {
            "id": conflict_id, "content": conflict_content,
            "type": "fact", "importance": 5, "tags": [],
            "created_at": now, "updated_at": now,
            "tenant_id": "local", "project_id": "test-ns",
        })
        svc.index.insert_with_embedding(
            memory_id=conflict_id, namespace="test-ns", tenant_id="local",
            type_="fact", importance=5, tags=[], content_hash=content_hash,
            storage_path=f"test-ns/{conflict_id}.yaml", content=conflict_content,
            source=None, metadata=None, created_at=now, updated_at=now,
        )

        ok = sync.resolve_conflict(conflict_id, "local", svc)
        assert ok is True
        # Conflict sibling should be deleted
        assert svc.get(conflict_id) is None
        # Original should still exist
        assert svc.get(original_id) is not None

    def test_resolve_conflict_keep_remote(self, svc, sync):
        sync.enable("test-ns", api_key="memra_live_test123")
        data, _ = svc.add({"content": "local version", "project_id": "test-ns"})
        original_id = data["id"]

        # Create conflict sibling
        conflict_id = f"{original_id}_conflict_remote"
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        import hashlib
        conflict_content = "remote version"
        content_hash = hashlib.sha256(conflict_content.encode()).hexdigest()
        svc.store.write("test-ns", conflict_id, {
            "id": conflict_id, "content": conflict_content,
            "type": "fact", "importance": 5, "tags": [],
            "created_at": now, "updated_at": now,
            "tenant_id": "local", "project_id": "test-ns",
        })
        svc.index.insert_with_embedding(
            memory_id=conflict_id, namespace="test-ns", tenant_id="local",
            type_="fact", importance=5, tags=[], content_hash=content_hash,
            storage_path=f"test-ns/{conflict_id}.yaml", content=conflict_content,
            source=None, metadata=None, created_at=now, updated_at=now,
        )

        ok = sync.resolve_conflict(conflict_id, "remote", svc)
        assert ok is True
        # Original should now have remote content
        original = svc.get(original_id)
        assert original is not None
        assert original["content"] == "remote version"
        # Conflict sibling should be deleted
        assert svc.get(conflict_id) is None


# -------------------------------------------------------------------
# API key persistence
# -------------------------------------------------------------------

class TestApiKeyPersistence:
    def test_api_key_loaded_by_new_service_instance(self, svc):
        """enable() persists the key; a fresh SyncService (new process) loads it.

        Without persistence the key only lived in the enabling process's
        memory, so any CLI push/pull in another process sent an empty Bearer
        token and always got a 401.
        """
        sync = svc.sync_service
        sync.enable("my-ns", api_key="memra_live_persisted123")

        creds_path = sync._credentials_path()
        assert creds_path.exists()
        # Secret material -- file must be owner-only
        assert (creds_path.stat().st_mode & 0o777) == 0o600

        fresh = SyncService(index=svc.index)
        assert fresh._api_keys.get("my-ns") == "memra_live_persisted123"

    def test_disable_removes_persisted_key(self, svc):
        sync = svc.sync_service
        sync.enable("my-ns", api_key="memra_live_persisted123")
        sync.disable("my-ns")

        fresh = SyncService(index=svc.index)
        assert fresh._api_keys.get("my-ns") is None


# -------------------------------------------------------------------
# Account tier detection (reads GET /usage)
# -------------------------------------------------------------------

def _usage_response(tier):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"tier": tier}
    resp.raise_for_status = MagicMock()
    return resp


class TestCheckAccountTier:
    def test_reads_tier_from_usage_endpoint(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1")
        with patch("httpx.get", return_value=_usage_response("admin")) as mock_get:
            assert sync.check_account_tier("my-ns") == "admin"
        assert "/usage" in mock_get.call_args[0][0]

    def test_result_is_cached(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1")
        with patch("httpx.get", return_value=_usage_response("team")) as mock_get:
            assert sync.check_account_tier("my-ns") == "team"
            assert sync.check_account_tier("my-ns") == "team"
        mock_get.assert_called_once()  # second call served from cache

    def test_unknown_namespace_returns_none(self, sync):
        with patch("httpx.get") as mock_get:
            assert sync.check_account_tier("never-enabled") is None
        mock_get.assert_not_called()  # fail-closed before any network call

    def test_http_error_is_fail_closed(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1")
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert sync.check_account_tier("my-ns") is None

    def test_missing_tier_field_returns_none(self, sync):
        sync.enable("my-ns", api_key="memra_live_test123", api_url="https://example.com/api/v1")
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {}  # no tier key
        resp.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=resp):
            assert sync.check_account_tier("my-ns") is None
