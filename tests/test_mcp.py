"""Tests for MCP server tool registration and functionality."""

from __future__ import annotations

import pytest
from pathlib import Path


# ------------------------------------------------------------------
# Factory tests
# ------------------------------------------------------------------
class TestServiceFactory:
    """Tests for create_service() factory function."""

    def test_create_service_returns_memory_service(self, tmp_path: Path):
        from memra_local.services.factory import create_service
        from memra_local.services.memory_service import MemoryService

        svc = create_service(storage_dir=tmp_path)
        assert isinstance(svc, MemoryService)
        assert svc.store is not None
        assert svc.index is not None
        assert svc.embedding_service is not None

    def test_create_service_global_scope(self, tmp_path: Path, monkeypatch):
        from memra_local.services.factory import create_service

        # Monkeypatch home to use tmp_path
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        svc = create_service(scope="global", storage_dir=tmp_path / ".memra" / "global")
        assert svc is not None

    def test_create_service_with_storage_dir(self, tmp_path: Path):
        from memra_local.services.factory import create_service

        custom_dir = tmp_path / "custom_storage"
        svc = create_service(storage_dir=custom_dir)
        assert custom_dir.exists()
        assert (custom_dir / ".gitignore").exists()


# ------------------------------------------------------------------
# MCP tool registration tests
# ------------------------------------------------------------------
class TestMCPToolRegistration:
    """Tests for MCP tool registration."""

    def test_all_tools_registered(self):
        from memra_local.mcp_server import mcp_app

        # FastMCP._tool_manager._tools is a dict of registered tools
        tools = mcp_app._tool_manager._tools
        assert len(tools) == 18, f"Expected 18 tools, got {len(tools)}: {list(tools.keys())}"

    def test_tool_names_match_cloud(self):
        from memra_local.mcp_server import mcp_app

        tool_names = set(mcp_app._tool_manager._tools.keys())
        expected = {
            "memra_add",
            "memra_get",
            "memra_search",
            "memra_list",
            "memra_bootstrap",
            "memra_get_health",
            "memra_add_decision",
            "memra_add_pattern",
            "memra_refresh_memory",
            "memra_promote",
            "memra_supersede",
            "memra_history",
            "memra_sync_enable",
            "memra_sync_push",
            "memra_sync_pull",
            "memra_migrate",
            "memra_remember",
            "memra_recall",
        }
        assert tool_names == expected, f"Missing: {expected - tool_names}, Extra: {tool_names - expected}"


# ------------------------------------------------------------------
# MCP tool functionality tests
# ------------------------------------------------------------------
class TestMCPToolFunctionality:
    """Tests for MCP tool execution via direct function calls."""

    @pytest.fixture(autouse=True)
    def setup_service(self, tmp_path: Path):
        """Initialize MCP service with tmp_path storage."""
        import memra_local.mcp_server as mcp_mod
        from memra_local.services.factory import create_service

        svc = create_service(storage_dir=tmp_path)
        mcp_mod._service = svc
        mcp_mod._scope = "project"
        yield
        mcp_mod._service = None

    def test_memra_add_returns_id(self):
        from memra_local.mcp_server import memra_add

        result = memra_add(content="Test memory", namespace="test-ns")
        assert "id" in result
        assert result["id"].startswith("mem_")

    def test_memra_search_returns_memories(self):
        from memra_local.mcp_server import memra_add, memra_search

        memra_add(content="Python is a programming language", namespace="test-ns")
        result = memra_search(query="programming", namespace="test-ns")
        assert "memories" in result

    def test_memra_get_returns_memory(self):
        from memra_local.mcp_server import memra_add, memra_get

        added = memra_add(content="Get me later", namespace="test-ns")
        memory_id = added["id"]
        result = memra_get(memory_id=memory_id)
        assert result["id"] == memory_id
        assert result["content"] == "Get me later"

    def test_memra_get_not_found(self):
        from memra_local.mcp_server import memra_get

        result = memra_get(memory_id="mem_nonexistent")
        assert "error" in result

    def test_memra_list_returns_paginated(self):
        from memra_local.mcp_server import memra_add, memra_list

        memra_add(content="List item 1", namespace="test-ns")
        memra_add(content="List item 2", namespace="test-ns")
        result = memra_list(namespace="test-ns")
        assert "memories" in result
        assert "total" in result
        assert result["total"] >= 2

    def test_memra_bootstrap_returns_memories(self):
        from memra_local.mcp_server import memra_add, memra_bootstrap

        memra_add(content="Important fact", namespace="test-ns", importance=9)
        result = memra_bootstrap(namespace="test-ns")
        assert "memories" in result
        assert len(result["memories"]) >= 1

    def test_memra_get_health_returns_stats(self):
        from memra_local.mcp_server import memra_add, memra_get_health

        memra_add(content="Health check memory", namespace="test-ns")
        result = memra_get_health(namespace="test-ns")
        assert "memory_count" in result
        assert "scope" in result
        assert result["memory_count"] >= 1

    def test_memra_add_decision_sets_type_and_importance(self):
        from memra_local.mcp_server import memra_add_decision, memra_get

        result = memra_add_decision(
            content="Use PostgreSQL for metadata",
            namespace="test-ns",
            context="Evaluating databases",
        )
        assert "id" in result
        # Verify stored with correct type and importance
        memory = memra_get(memory_id=result["id"])
        assert memory["type"] == "decision"
        assert memory["importance"] == 8

    def test_memra_add_pattern_formats_steps(self):
        from memra_local.mcp_server import memra_add_pattern, memra_get

        result = memra_add_pattern(
            name="Deploy workflow",
            namespace="test-ns",
            steps=["Build image", "Push to registry", "Deploy"],
            gotchas=["Check env vars"],
        )
        assert "id" in result
        memory = memra_get(memory_id=result["id"])
        assert memory["type"] == "pattern"
        assert "Deploy workflow" in memory["content"]
        assert "Build image" in memory["content"]
        assert "Check env vars" in memory["content"]

    def test_memra_refresh_memory_updates_timestamp(self):
        import time
        from memra_local.mcp_server import memra_add, memra_refresh_memory

        added = memra_add(content="Refresh me", namespace="test-ns")
        original_updated = added.get("updated_at")
        time.sleep(1.1)  # timestamps are second-granularity; ensure tick
        result = memra_refresh_memory(memory_id=added["id"])
        assert "id" in result
        assert result["updated_at"] != original_updated

    def test_memra_refresh_memory_not_found(self):
        from memra_local.mcp_server import memra_refresh_memory

        result = memra_refresh_memory(memory_id="mem_nonexistent")
        assert "error" in result

    def test_mcp_supersede(self):
        """memra_supersede returns new_memory and superseded_memory_id."""
        from memra_local.mcp_server import memra_add, memra_supersede

        added = memra_add(content="Old content", namespace="test-ns")
        old_id = added["id"]

        result = memra_supersede(old_memory_id=old_id, content="New content")
        assert "new_memory" in result
        assert result["new_memory"]["id"].startswith("mem_")
        assert result["superseded_memory_id"] == old_id

    def test_mcp_supersede_error(self):
        """memra_supersede with bad id returns error dict."""
        from memra_local.mcp_server import memra_supersede

        result = memra_supersede(old_memory_id="mem_nonexistent", content="New")
        assert "error" in result

    def test_mcp_history(self):
        """memra_history returns chain list and length."""
        from memra_local.mcp_server import memra_add, memra_supersede, memra_history

        added = memra_add(content="Original", namespace="test-ns")
        old_id = added["id"]
        memra_supersede(old_memory_id=old_id, content="Updated")

        result = memra_history(memory_id=old_id)
        assert "chain" in result
        assert "length" in result
        assert result["length"] == 2
        assert len(result["chain"]) == 2

    def test_mcp_history_not_found(self):
        """memra_history with bad id returns error dict."""
        from memra_local.mcp_server import memra_history

        result = memra_history(memory_id="mem_nonexistent")
        assert "error" in result

    def test_memra_sync_enable_returns_status(self):
        from memra_local.mcp_server import memra_sync_enable

        result = memra_sync_enable(
            namespace="test-sync-ns",
            api_key="memra_live_test123",
            api_url="https://example.com/api/v1",
        )
        assert result["status"] == "enabled"
        assert result["namespace"] == "test-sync-ns"

    # --------------------------------------------------------------
    # v4.3 canonical verbs: memra_remember + memra_recall
    # --------------------------------------------------------------

    def test_memra_remember_creates_fact(self):
        from memra_local.mcp_server import memra_remember, memra_get

        result = memra_remember(content="Remember this", namespace="test-ns")
        assert "id" in result
        stored = memra_get(memory_id=result["id"])
        assert stored["type"] == "fact"

    def test_memra_remember_entries_bulk(self):
        from memra_local.mcp_server import memra_remember

        result = memra_remember(
            entries=[
                {"content": "First", "namespace": "test-ns"},
                {"content": "Second", "namespace": "test-ns"},
            ]
        )
        assert result["count"] == 2
        assert len(result["memories"]) == 2

    def test_memra_remember_decision_dispatches(self):
        from memra_local.mcp_server import memra_remember, memra_get

        result = memra_remember(
            type="decision",
            content="Use Postgres",
            namespace="test-ns",
            context="DB selection",
        )
        stored = memra_get(memory_id=result["id"])
        assert stored["type"] == "decision"
        assert stored["importance"] == 8

    def test_memra_remember_pattern_dispatches(self):
        from memra_local.mcp_server import memra_remember, memra_get

        result = memra_remember(
            type="pattern",
            title="Deploy flow",
            steps=["build", "push"],
            namespace="test-ns",
        )
        stored = memra_get(memory_id=result["id"])
        assert stored["type"] == "pattern"
        assert "Deploy flow" in stored["content"]

    def test_memra_recall_matches_search(self):
        from memra_local.mcp_server import memra_remember, memra_recall

        memra_remember(content="Python is a programming language", namespace="test-ns")
        result = memra_recall(query="programming", namespace="test-ns")
        assert "memories" in result
        assert "meta" in result
