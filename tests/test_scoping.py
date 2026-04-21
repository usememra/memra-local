"""Tests for scope resolution and config loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from memra_local.config import detect_scope, load_config, resolve_storage_dir


class TestResolveStorageDir:
    def test_global_scope(self) -> None:
        result = resolve_storage_dir("global")
        assert result == Path.home() / ".memra" / "global"

    def test_project_scope(self, tmp_storage: Path) -> None:
        result = resolve_storage_dir("project", cwd=tmp_storage)
        assert result == tmp_storage / ".memra"

    def test_unknown_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scope"):
            resolve_storage_dir("invalid_scope")


class TestDetectScope:
    def test_auto_detect_with_memra_dir(self, tmp_project_dir: Path) -> None:
        scope = detect_scope(cwd=tmp_project_dir)
        assert scope == "project"

    def test_auto_detect_without_memra_dir(self, tmp_storage: Path) -> None:
        scope = detect_scope(cwd=tmp_storage)
        assert scope == "global"


class TestMemoryModel:
    def test_memory_model_fields(self) -> None:
        from memra_local.models import Memory

        now = "2026-04-08T00:00:00Z"
        m = Memory(
            id="test-id",
            content="hello",
            tenant_id="local",
            project_id="default",
            type="fact",
            importance=5,
            created_at=now,
            updated_at=now,
        )
        assert m.id == "test-id"
        assert m.content == "hello"
        assert m.tenant_id == "local"
        assert m.project_id == "default"
        assert m.type == "fact"
        assert m.importance == 5
        assert m.tags == []
        assert m.source is None
        assert m.metadata is None
        assert m.embedding_status == "none"
        assert m.expires_at is None
        assert m.confidence == 1.0
        assert m.staleness_score == 0
        assert m.last_accessed_at is None
        assert m.status == "active"
        assert m.superseded_by is None
        assert m.ttl_days is None
        assert m.soft_ttl_days is None
        assert m.created_at == now
        assert m.updated_at == now

    def test_add_memory_request_defaults(self) -> None:
        from memra_local.models import AddMemoryRequest

        req = AddMemoryRequest(content="test")
        assert req.tenant_id == "local"
        assert req.project_id == "default"
        assert req.type == "fact"
        assert req.importance == 5
        assert req.tags == []
        assert req.source is None
        assert req.metadata is None


class TestLoadConfig:
    def test_load_config_defaults(self) -> None:
        config = load_config(config_path=Path("/nonexistent/path/config.yaml"))
        assert config["port"] == 8765
        assert config["host"] == "127.0.0.1"
        assert config["scope"] == "auto"
