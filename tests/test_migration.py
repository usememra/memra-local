"""Tests for migration service: local->cloud batch upload."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from memra_local.services.factory import create_service
from memra_local.services.migration_service import MigrationService
from memra_local.models import MigrateResult


@pytest.fixture
def tmp_storage():
    d = Path(tempfile.mkdtemp(prefix="memra_migrate_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def svc(tmp_storage):
    """Create a MemoryService with test storage."""
    return create_service(scope="global", storage_dir=tmp_storage)


@pytest.fixture
def populated_svc(svc):
    """Service with 5 test memories added."""
    for i in range(5):
        svc.add({
            "content": f"Test memory number {i}",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 5 + (i % 3),
            "tags": ["test"],
            "source": "unit-test",
        })
    return svc


def _mock_batch_response(created: int, duplicates: int = 0):
    """Create a mock httpx.Response for batch endpoint."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "created": created,
        "duplicates": duplicates,
        "memories": [],
    }
    resp.raise_for_status = MagicMock()
    return resp


# -------------------------------------------------------------------
# Basic migration
# -------------------------------------------------------------------

class TestMigration:
    def test_migrate_uploads_all_memories(self, populated_svc):
        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        with patch("httpx.post", return_value=_mock_batch_response(5)) as mock_post:
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_test",
            )

        assert result.total == 5
        assert result.migrated == 5
        assert result.failed == 0
        assert result.dry_run is False
        mock_post.assert_called_once()
        # Verify POST was to batch endpoint
        call_args = mock_post.call_args
        assert "memories/batch" in call_args[0][0] or "memories/batch" in str(call_args)

    def test_migrate_batches_in_groups_of_50(self, tmp_storage):
        """With 120 memories, should make 3 batch calls (50+50+20)."""
        svc = create_service(scope="global", storage_dir=tmp_storage)
        for i in range(120):
            svc.add({
                "content": f"Batch test memory {i}",
                "project_id": "batch-ns",
                "type": "fact",
                "importance": 5,
            })

        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        def side_effect(*args, **kwargs):
            body = kwargs.get("json", {}) or (json.loads(args[1]) if len(args) > 1 else {})
            if isinstance(body, dict):
                count = len(body.get("memories", []))
            else:
                count = 0
            return _mock_batch_response(count)

        with patch("httpx.post", side_effect=side_effect) as mock_post:
            result = migration.migrate(index=svc.index, store=svc.store, project_id="proj_test")

        assert mock_post.call_count == 3  # 50 + 50 + 20
        assert result.migrated == 120
        assert result.total == 120


# -------------------------------------------------------------------
# PII masking
# -------------------------------------------------------------------

class TestMigrationPii:
    def test_migrate_masks_content_via_pii_client(self, populated_svc):
        mock_pii = MagicMock()
        mock_pii.mask_batch.return_value = [
            f"[MASKED] memory {i}" for i in range(5)
        ]

        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
            pii_client=mock_pii,
        )

        with patch("httpx.post", return_value=_mock_batch_response(5)) as mock_post:
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_test",
            )

        assert result.migrated == 5
        mock_pii.mask_batch.assert_called_once()
        # Verify the POST body has masked content
        call_kwargs = mock_post.call_args[1] if mock_post.call_args[1] else {}
        if "json" in call_kwargs:
            memories = call_kwargs["json"]["memories"]
            assert all("[MASKED]" in m["content"] for m in memories)

    def test_migrate_records_error_when_pii_fails(self, populated_svc):
        mock_pii = MagicMock()
        mock_pii.mask_batch.return_value = None  # Failure

        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
            pii_client=mock_pii,
        )

        with patch("httpx.post") as mock_post:
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_test",
            )

        assert result.failed == 5
        assert len(result.errors) >= 1
        mock_post.assert_not_called()  # Should not upload unmasked data


# -------------------------------------------------------------------
# Idempotent (duplicates)
# -------------------------------------------------------------------

class TestMigrationIdempotent:
    def test_migrate_skips_duplicates(self, populated_svc):
        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        with patch("httpx.post", return_value=_mock_batch_response(created=2, duplicates=3)):
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_test",
            )

        assert result.migrated == 2
        assert result.skipped == 3
        assert result.total == 5


# -------------------------------------------------------------------
# Dry run
# -------------------------------------------------------------------

class TestMigrationDryRun:
    def test_dry_run_returns_count_without_uploading(self, populated_svc):
        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        with patch("httpx.post") as mock_post:
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_test",
                dry_run=True,
            )

        assert result.dry_run is True
        assert result.total == 5
        assert result.migrated == 0
        mock_post.assert_not_called()


# -------------------------------------------------------------------
# Partial failure
# -------------------------------------------------------------------

class TestMigrationPartialFailure:
    def test_partial_failure_continues_and_reports(self, tmp_storage):
        """First batch succeeds, second fails -- migration continues."""
        svc = create_service(scope="global", storage_dir=tmp_storage)
        for i in range(75):
            svc.add({
                "content": f"Partial fail test {i}",
                "project_id": "fail-ns",
                "type": "fact",
                "importance": 5,
            })

        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ConnectError("Connection refused")
            body = kwargs.get("json", {})
            count = len(body.get("memories", []))
            return _mock_batch_response(count)

        with patch("httpx.post", side_effect=side_effect):
            result = migration.migrate(index=svc.index, store=svc.store, project_id="proj_test")

        # 75 memories = batch 1 (50) + batch 2 (25). Batch 2 fails.
        assert result.migrated == 50
        assert result.failed == 25
        assert len(result.errors) >= 1
        assert result.total == 75


# -------------------------------------------------------------------
# Progress callback
# -------------------------------------------------------------------

class TestMigrationProgress:
    def test_progress_callback_called(self, populated_svc):
        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        progress_calls = []

        def on_progress(migrated, total):
            progress_calls.append((migrated, total))

        with patch("httpx.post", return_value=_mock_batch_response(5)):
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_test",
                progress_callback=on_progress,
            )

        assert len(progress_calls) >= 1
        # Last call should have the final count
        last_migrated, last_total = progress_calls[-1]
        assert last_total == 5


# -------------------------------------------------------------------
# MigrateResult model
# -------------------------------------------------------------------

class TestMigrateResult:
    def test_migrate_result_fields(self):
        r = MigrateResult(
            total=100, migrated=80, skipped=15, failed=5,
            errors=["batch 3 failed"], dry_run=False,
        )
        assert r.total == 100
        assert r.migrated == 80
        assert r.skipped == 15
        assert r.failed == 5
        assert r.errors == ["batch 3 failed"]
        assert r.dry_run is False

    def test_migrate_result_defaults(self):
        r = MigrateResult(total=10, migrated=10, skipped=0, failed=0)
        assert r.errors == []
        assert r.dry_run is False


# -------------------------------------------------------------------
# Cloud batch contract (regression: 422 + silent-207)
# -------------------------------------------------------------------

class TestMigrationCloudContract:
    def test_payload_carries_tenant_id_and_target_project(self, populated_svc):
        """Each memory must send tenant_id (=namespace) and the target project_id.

        Regression for the 422 'tenant id required' / 'project not found': the
        cloud batch endpoint requires both per memory, and project_id must be a
        real cloud project, not the local namespace.
        """
        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        with patch("httpx.post", return_value=_mock_batch_response(5)) as mock_post:
            migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_target_abc",
            )

        sent = mock_post.call_args.kwargs["json"]["memories"]
        assert sent, "no memories sent"
        for m in sent:
            assert m["project_id"] == "proj_target_abc"
            assert m["tenant_id"] == "test-ns"  # local namespace preserved

    def test_207_per_item_failures_are_counted_not_swallowed(self, populated_svc):
        """A 207 with per-item errors must increment failed and surface a message.

        raise_for_status() does not raise on 207, so without explicit handling
        the failures vanish as a silent '0 migrated'.
        """
        resp = MagicMock()
        resp.status_code = 207
        resp.json.return_value = {
            "total": 5,
            "created": 2,
            "duplicates": 0,
            "errors": 3,
            "results": [
                {"index": 0, "status": "created"},
                {"index": 1, "status": "created"},
                {"index": 2, "status": "error",
                 "error": {"code": "project_not_found",
                           "message": "No project with ID proj_x exists in your account"}},
                {"index": 3, "status": "error", "error": {"code": "internal_error", "message": "boom"}},
                {"index": 4, "status": "error", "error": {"code": "internal_error", "message": "boom"}},
            ],
        }
        resp.raise_for_status = MagicMock()

        migration = MigrationService(
            api_url="https://example.com/api/v1",
            api_key="memra_live_test123",
        )

        with patch("httpx.post", return_value=resp):
            result = migration.migrate(
                index=populated_svc.index,
                store=populated_svc.store,
                project_id="proj_x",
            )

        assert result.migrated == 2
        assert result.failed == 3
        assert any("No project with ID" in e for e in result.errors)
