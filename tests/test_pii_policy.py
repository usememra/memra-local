"""Tests for PII sharing modes in sync: local_private, shared_masked, shared_raw."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from memra_local.services.factory import create_service
from memra_local.services.pii_client import PiiClient
from memra_local.services.sync_service import SyncService


@pytest.fixture
def tmp_storage():
    d = Path(tempfile.mkdtemp(prefix="memra_pii_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def svc(tmp_storage):
    return create_service(scope="global", storage_dir=tmp_storage)


@pytest.fixture
def sync(svc):
    return svc.sync_service


# -------------------------------------------------------------------
# PiiClient
# -------------------------------------------------------------------


class TestPiiClient:
    def test_mask_batch_returns_masked_strings(self):
        client = PiiClient(api_url="https://example.com/api/v1", api_key="memra_live_test")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"masked_content": "[PII_PERSON_abc12345] lives in Berlin", "was_masked": True, "token_count": 1},
                {"masked_content": "No PII here", "was_masked": False, "token_count": 0},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = client.mask_batch(["John lives in Berlin", "No PII here"])

        assert result is not None
        assert len(result) == 2
        assert "[PII_PERSON_abc12345]" in result[0]
        assert result[1] == "No PII here"
        mock_post.assert_called_once()

    def test_mask_batch_returns_none_on_http_error(self):
        client = PiiClient(api_url="https://example.com/api/v1", api_key="memra_live_test")

        with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
            result = client.mask_batch(["John lives in Berlin"])

        assert result is None

    def test_mask_batch_returns_none_on_server_error(self):
        client = PiiClient(api_url="https://example.com/api/v1", api_key="memra_live_test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )

        with patch("httpx.post", return_value=mock_resp):
            result = client.mask_batch(["John lives in Berlin"])

        assert result is None


# -------------------------------------------------------------------
# Local Private
# -------------------------------------------------------------------


class TestLocalPrivate:
    def test_local_private_push_returns_zero(self, sync):
        sync.enable("private-ns", api_key="memra_live_test", pii_mode="local_private")
        sync.record_event("memory_created", "private-ns", "mem_1", {"content": "secret"})
        result = sync.push("private-ns")
        assert result.pushed == 0
        assert "local_private" in (result.error or "")

    def test_enable_defaults_to_local_private(self, sync):
        sync.enable("my-ns", api_key="memra_live_test")
        mode = sync.get_mode("my-ns")
        assert mode == "local_private"


# -------------------------------------------------------------------
# Shared Masked
# -------------------------------------------------------------------


class TestSharedMasked:
    def test_shared_masked_calls_pii_client_before_push(self, sync):
        pii_client = MagicMock()
        pii_client.mask_batch.return_value = ["[MASKED] content"]
        sync._pii_client = pii_client

        sync.enable("masked-ns", api_key="memra_live_test", pii_mode="shared_masked")
        sync.record_event("memory_created", "masked-ns", "mem_1", {"content": "John Smith lives here"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cursor": 1, "pushed": 1, "conflicts": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = sync.push("masked-ns")

        assert result.pushed == 1
        pii_client.mask_batch.assert_called_once()

    def test_shared_masked_pii_failure_blocks_push(self, sync):
        pii_client = MagicMock()
        pii_client.mask_batch.return_value = None  # Failure
        sync._pii_client = pii_client

        sync.enable("masked-ns", api_key="memra_live_test", pii_mode="shared_masked")
        sync.record_event("memory_created", "masked-ns", "mem_1", {"content": "John Smith"})

        result = sync.push("masked-ns")
        assert result.pushed == 0
        assert "PII masking failed" in (result.error or "")


# -------------------------------------------------------------------
# Shared Raw
# -------------------------------------------------------------------


class TestSharedRaw:
    def test_shared_raw_without_team_tier_returns_error(self, sync):
        sync.enable("raw-ns", api_key="memra_live_test", pii_mode="shared_raw")
        sync.record_event("memory_created", "raw-ns", "mem_1", {"content": "raw data"})

        # Mock tier check to return "hobby"
        with patch.object(sync, "check_account_tier", return_value="hobby"):
            result = sync.push("raw-ns")

        assert result.pushed == 0
        assert "team" in (result.error or "").lower() or "tier" in (result.error or "").lower()

    def test_shared_raw_with_team_tier_succeeds(self, sync):
        sync.enable("raw-ns", api_key="memra_live_test", pii_mode="shared_raw")
        sync.record_event("memory_created", "raw-ns", "mem_1", {"content": "raw data"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cursor": 1, "pushed": 1, "conflicts": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(sync, "check_account_tier", return_value="team"):
            with patch("httpx.post", return_value=mock_response):
                result = sync.push("raw-ns")

        assert result.pushed == 1


# -------------------------------------------------------------------
# PII Failure
# -------------------------------------------------------------------


class TestPiiFailure:
    def test_pii_failure_never_sends_unmasked_data(self, sync):
        """When PII masking fails, no HTTP POST to cloud should happen."""
        pii_client = MagicMock()
        pii_client.mask_batch.return_value = None
        sync._pii_client = pii_client

        sync.enable("masked-ns", api_key="memra_live_test", pii_mode="shared_masked")
        sync.record_event("memory_created", "masked-ns", "mem_1", {"content": "sensitive PII"})

        with patch("httpx.post") as mock_post:
            result = sync.push("masked-ns")

        assert result.pushed == 0
        mock_post.assert_not_called()


# -------------------------------------------------------------------
# Set Mode / Get Mode
# -------------------------------------------------------------------


class TestSetMode:
    def test_set_mode_persists_in_db(self, sync):
        sync.enable("my-ns", api_key="memra_live_test")
        ok = sync.set_mode("my-ns", "shared_masked")
        assert ok is True
        assert sync.get_mode("my-ns") == "shared_masked"

    def test_set_mode_returns_false_for_unknown_namespace(self, sync):
        ok = sync.set_mode("nonexistent", "shared_masked")
        assert ok is False

    def test_get_mode_returns_none_for_unknown_namespace(self, sync):
        assert sync.get_mode("nonexistent") is None

    def test_set_mode_validates_modes(self, sync):
        sync.enable("my-ns", api_key="memra_live_test")
        # Valid modes should work
        for mode in ("local_private", "shared_masked", "shared_raw"):
            assert sync.set_mode("my-ns", mode) is True
            assert sync.get_mode("my-ns") == mode

    def test_set_mode_rejects_invalid_mode(self, sync):
        sync.enable("my-ns", api_key="memra_live_test")
        ok = sync.set_mode("my-ns", "invalid_mode")
        assert ok is False

    def test_existing_namespaces_default_to_shared_masked(self, svc):
        """Namespaces created before the pii_mode migration default to shared_masked."""
        # The SQLite migration sets DEFAULT 'shared_masked' for existing rows
        # Simulate an old-style enable (which now includes pii_mode column with default)
        conn = svc.sync_service._conn
        conn.execute(
            """INSERT OR REPLACE INTO sync_cursors
               (namespace, cloud_api_url, remote_cursor, last_synced_at, enabled_at)
               VALUES ('legacy-ns', 'https://usememra.com/api/v1', 0, NULL, '2026-01-01T00:00:00Z')""",
        )
        conn.commit()
        mode = svc.sync_service.get_mode("legacy-ns")
        assert mode == "shared_masked"
