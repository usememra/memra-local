"""Tests for cloud bootstrap merge in memra-local.

Phase 77-04: `bootstrap()` must merge local high-importance memories with
cloud-promoted memories when sync is enabled.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from memra_local.services.factory import create_service


@pytest.fixture
def tmp_storage():
    d = Path(tempfile.mkdtemp(prefix="memra_cloud_boot_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def svc(tmp_storage):
    return create_service(scope="global", storage_dir=tmp_storage)


@pytest.fixture
def tmp_home(monkeypatch):
    """Redirect ~/.memra/bootstrap-cache.json to a temp dir."""
    d = Path(tempfile.mkdtemp(prefix="memra_home_"))
    monkeypatch.setenv("HOME", str(d))
    # Pre-create ~/.memra so the cache writer doesn't have to.
    (d / ".memra").mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _add_local(svc, content: str, importance: int = 5, namespace: str = "proj_x") -> dict:
    mem, _ = svc.add({
        "content": content,
        "project_id": namespace,
        "tenant_id": "local",
        "importance": importance,
    })
    return mem


def _cloud_response(memories: list[dict]) -> dict:
    return {
        "data": {
            "agent_id": "agent_x",
            "tenant_id": "local",
            "project_id": "proj_x",
            "memories": memories,
            "token_estimate": 0,
            "health_warnings": [],
            "revision": 1,
            "generated_at": "2026-04-19T00:00:00Z",
            "cache_ttl": 60,
            "cached": False,
        }
    }


# -------------------------------------------------------------------
# Test 1: Sync enabled — merges cloud memories with local
# -------------------------------------------------------------------

class TestCloudBootstrapMerge:
    def test_merge_cloud_and_local(self, svc, tmp_home):
        _add_local(svc, "local A", importance=9)
        _add_local(svc, "local B", importance=7)

        svc.sync_service.enable(
            "proj_x",
            api_key="memra_live_test",
            agent_id="agent_x",
        )

        cloud_memories = [
            {"id": "mem_cloud1", "content": "cloud 1", "type": "fact", "importance": 8, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
            {"id": "mem_cloud2", "content": "cloud 2", "type": "fact", "importance": 6, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
            {"id": "mem_cloud3", "content": "cloud 3", "type": "fact", "importance": 5, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
        ]

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: _cloud_response(cloud_memories),
                raise_for_status=lambda: None,
            )
            results = svc.bootstrap("proj_x", "local", limit=20)

        ids = [m["id"] for m in results]
        # Local memories present
        assert any(m["content"] == "local A" for m in results)
        assert any(m["content"] == "local B" for m in results)
        # All 3 cloud memories present
        assert "mem_cloud1" in ids
        assert "mem_cloud2" in ids
        assert "mem_cloud3" in ids
        # HTTP was called with the agent_id path and auth header
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "/v1/agents/agent_x/bootstrap" in call_url

    def test_cloud_wins_on_id_collision(self, svc, tmp_home):
        local = _add_local(svc, "local content", importance=5)
        svc.sync_service.enable("proj_x", api_key="k", agent_id="agent_x")

        cloud_memories = [
            {"id": local["id"], "content": "cloud override", "type": "fact", "importance": 9, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
        ]

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: _cloud_response(cloud_memories),
                raise_for_status=lambda: None,
            )
            results = svc.bootstrap("proj_x", "local", limit=20)

        # Cloud version of the memory should win
        matching = [m for m in results if m["id"] == local["id"]]
        assert len(matching) == 1
        assert matching[0]["content"] == "cloud override"

    def test_cloud_failure_degrades_to_local(self, svc, tmp_home, caplog):
        _add_local(svc, "local only", importance=9)
        svc.sync_service.enable("proj_x", api_key="k", agent_id="agent_x")

        with patch(
            "memra_local.services.sync_service.httpx.get",
            side_effect=httpx.ConnectError("boom"),
        ):
            with caplog.at_level("WARNING"):
                results = svc.bootstrap("proj_x", "local", limit=20)

        assert any(m["content"] == "local only" for m in results)
        # No exception propagated; warning logged.
        assert any("cloud bootstrap" in rec.message.lower() for rec in caplog.records)

    def test_cloud_http_error_degrades_to_local(self, svc, tmp_home):
        _add_local(svc, "local only", importance=9)
        svc.sync_service.enable("proj_x", api_key="k", agent_id="agent_x")

        bad_resp = MagicMock()
        bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        )

        with patch("memra_local.services.sync_service.httpx.get", return_value=bad_resp):
            results = svc.bootstrap("proj_x", "local", limit=20)

        assert any(m["content"] == "local only" for m in results)
        # No cloud memories
        assert all(not m["id"].startswith("mem_cloud") for m in results)


# -------------------------------------------------------------------
# Test 2: Sync disabled — no HTTP call
# -------------------------------------------------------------------

class TestCloudBootstrapDisabled:
    def test_no_http_when_sync_disabled(self, svc, tmp_home):
        _add_local(svc, "local only", importance=9)

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            results = svc.bootstrap("proj_x", "local", limit=20)

        mock_get.assert_not_called()
        assert any(m["content"] == "local only" for m in results)


# -------------------------------------------------------------------
# Test 3: TTL cache prevents repeated HTTP
# -------------------------------------------------------------------

class TestCloudBootstrapCache:
    def test_cache_prevents_repeat_http(self, svc, tmp_home, monkeypatch):
        monkeypatch.setenv("MEMRA_BOOTSTRAP_TTL", "60")
        svc.sync_service.enable("proj_x", api_key="k", agent_id="agent_x")

        cloud_memories = [
            {"id": "mem_cloudA", "content": "cloud A", "type": "fact", "importance": 8, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
        ]

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: _cloud_response(cloud_memories),
                raise_for_status=lambda: None,
            )
            r1 = svc.bootstrap("proj_x", "local", limit=20)
            r2 = svc.bootstrap("proj_x", "local", limit=20)

        assert mock_get.call_count == 1
        ids1 = [m["id"] for m in r1]
        ids2 = [m["id"] for m in r2]
        assert "mem_cloudA" in ids1
        assert "mem_cloudA" in ids2

    def test_cache_expires_after_ttl(self, svc, tmp_home, monkeypatch):
        # TTL of 0 => immediately expired
        monkeypatch.setenv("MEMRA_BOOTSTRAP_TTL", "0")
        svc.sync_service.enable("proj_x", api_key="k", agent_id="agent_x")

        cloud_memories = [
            {"id": "mem_cloudA", "content": "cloud A", "type": "fact", "importance": 8, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
        ]

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: _cloud_response(cloud_memories),
                raise_for_status=lambda: None,
            )
            svc.bootstrap("proj_x", "local", limit=20)
            svc.bootstrap("proj_x", "local", limit=20)

        # Zero TTL means every call re-fetches.
        assert mock_get.call_count == 2

    def test_cache_file_written(self, svc, tmp_home):
        svc.sync_service.enable("proj_x", api_key="k", agent_id="agent_x")

        cloud_memories = [
            {"id": "mem_cloudA", "content": "cloud A", "type": "fact", "importance": 8, "tags": [], "created_at": "2026-04-18T00:00:00Z"},
        ]

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: _cloud_response(cloud_memories),
                raise_for_status=lambda: None,
            )
            svc.bootstrap("proj_x", "local", limit=20)

        cache_path = tmp_home / ".memra" / "bootstrap-cache.json"
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        # Keyed by (namespace|agent_id|tenant_id)
        key = "proj_x|agent_x|local"
        assert key in data
        assert data[key]["memories"][0]["id"] == "mem_cloudA"
        assert "fetched_at" in data[key]


# -------------------------------------------------------------------
# Test 4: No agent_id stored — cloud call skipped gracefully
# -------------------------------------------------------------------

class TestMissingAgentId:
    def test_enable_without_agent_id_skips_cloud(self, svc, tmp_home):
        _add_local(svc, "local only", importance=9)
        svc.sync_service.enable("proj_x", api_key="k")  # no agent_id

        with patch("memra_local.services.sync_service.httpx.get") as mock_get:
            results = svc.bootstrap("proj_x", "local", limit=20)

        mock_get.assert_not_called()
        assert any(m["content"] == "local only" for m in results)


# -------------------------------------------------------------------
# Test 5: Importance ordering happens in SQL, before LIMIT
# -------------------------------------------------------------------

class TestBootstrapImportanceOrdering:
    def test_old_high_importance_memory_survives_limit(self, svc):
        """An importance-10 memory older than 30 importance-1 rows must be in
        bootstrap. Regression: the index query ordered by created_at DESC and
        applied LIMIT before the importance sort, dropping old important rows.
        """
        top = _add_local(svc, "critical old fact", importance=10)
        # Make it strictly older than all the noise rows
        svc.index.update_by_id(top["id"], {"created_at": "2020-01-01T00:00:00Z"})

        for i in range(30):
            _add_local(svc, f"low-importance noise {i}", importance=1)

        results = svc.bootstrap("proj_x", "local")  # default limit=20
        ids = [m["id"] for m in results]
        assert top["id"] in ids
        # Highest importance comes first
        assert results[0]["id"] == top["id"]
