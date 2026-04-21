"""Tests for auto-capture hooks: installer, extractor, and CLI integration."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Installer tests (LOCAL-05a, LOCAL-05b, LOCAL-05c)
# ---------------------------------------------------------------------------

class TestInstaller:
    """Tests for hooks.installer module."""

    def test_install_hook(self, tmp_path: Path):
        """LOCAL-05a: install_session_end_hook creates settings.json with hook entry."""
        from memra_local.hooks.installer import install_session_end_hook

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"

        result = install_session_end_hook(settings_path=settings_path)

        assert result is True
        assert settings_path.exists()

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        session_end = settings["hooks"]["SessionEnd"]
        assert len(session_end) == 1

        hook_cmd = session_end[0]["hooks"][0]["command"]
        assert "memra-auto-capture" in hook_cmd
        assert session_end[0]["hooks"][0]["timeout"] == 30

    def test_install_idempotent(self, tmp_path: Path):
        """LOCAL-05b: calling install twice returns False, no duplicate entries."""
        from memra_local.hooks.installer import install_session_end_hook

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"

        first = install_session_end_hook(settings_path=settings_path)
        second = install_session_end_hook(settings_path=settings_path)

        assert first is True
        assert second is False

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        session_end = settings["hooks"]["SessionEnd"]
        assert len(session_end) == 1

    def test_uninstall_hook(self, tmp_path: Path):
        """LOCAL-05c: uninstall removes only Memra entry, preserves others."""
        from memra_local.hooks.installer import (
            install_session_end_hook,
            uninstall_session_end_hook,
        )

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"

        # Pre-populate with another hook
        other_hook = {
            "hooks": {
                "SessionEnd": [
                    {
                        "hooks": [
                            {"type": "command", "command": "other-tool cleanup", "timeout": 10}
                        ]
                    }
                ]
            }
        }
        settings_path.write_text(json.dumps(other_hook), encoding="utf-8")

        # Install memra hook
        install_session_end_hook(settings_path=settings_path)
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        assert len(settings["hooks"]["SessionEnd"]) == 2

        # Uninstall memra hook
        result = uninstall_session_end_hook(settings_path=settings_path)
        assert result is True

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        session_end = settings["hooks"]["SessionEnd"]
        assert len(session_end) == 1
        assert "other-tool" in session_end[0]["hooks"][0]["command"]

    def test_uninstall_not_installed(self, tmp_path: Path):
        """Uninstall returns False when no Memra hook exists."""
        from memra_local.hooks.installer import uninstall_session_end_hook

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        result = uninstall_session_end_hook(settings_path=settings_path)
        assert result is False

    def test_install_preserves_existing(self, tmp_path: Path):
        """Existing hooks/settings are not modified by install."""
        from memra_local.hooks.installer import install_session_end_hook

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"

        existing = {
            "permissions": {"allow": ["Edit"]},
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "lint check", "timeout": 5}]}
                ]
            },
        }
        settings_path.write_text(json.dumps(existing), encoding="utf-8")

        install_session_end_hook(settings_path=settings_path)

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        # Existing keys preserved
        assert settings["permissions"] == {"allow": ["Edit"]}
        assert len(settings["hooks"]["PreToolUse"]) == 1
        assert len(settings["hooks"]["SessionEnd"]) == 1


# ---------------------------------------------------------------------------
# Extractor tests (LOCAL-05d through LOCAL-05g)
# ---------------------------------------------------------------------------

def _make_transcript_jsonl(tmp_path: Path, entries: list[dict]) -> Path:
    """Helper: write a JSONL transcript file."""
    path = tmp_path / "transcript.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


class TestExtractor:
    """Tests for hooks.extractor module."""

    def test_parse_transcript(self, tmp_path: Path):
        """parse_transcript reads JSONL, returns assistant text strings only."""
        from memra_local.hooks.extractor import parse_transcript

        entries = [
            {"type": "user", "message": {"content": "hello"}},
            {"type": "assistant", "message": {"content": "I decided to use atomic writes for safety."}},
            {"type": "system", "message": {"content": "system msg"}},
            {"type": "assistant", "message": {"content": "Another assistant message."}},
        ]
        path = _make_transcript_jsonl(tmp_path, entries)
        result = parse_transcript(path)

        assert len(result) == 2
        assert "atomic writes" in result[0]
        assert "Another assistant" in result[1]

    def test_parse_transcript_content_blocks(self, tmp_path: Path):
        """Handles content as list of {type: text, text: ...} blocks."""
        from memra_local.hooks.extractor import parse_transcript

        entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Part one of the response."},
                        {"type": "tool_use", "id": "abc"},
                        {"type": "text", "text": "Part two."},
                    ]
                },
            },
        ]
        path = _make_transcript_jsonl(tmp_path, entries)
        result = parse_transcript(path)

        assert len(result) == 1
        assert "Part one" in result[0]
        assert "Part two" in result[0]

    def test_parse_transcript_skips_meta(self, tmp_path: Path):
        """Entries with isMeta: true are skipped."""
        from memra_local.hooks.extractor import parse_transcript

        entries = [
            {"type": "assistant", "isMeta": True, "message": {"content": "meta msg"}},
            {"type": "assistant", "message": {"content": "real msg"}},
        ]
        path = _make_transcript_jsonl(tmp_path, entries)
        result = parse_transcript(path)

        assert len(result) == 1
        assert "real msg" in result[0]

    def test_extract_decisions(self):
        """extract_from_text finds decision patterns."""
        from memra_local.hooks.extractor import extract_from_text

        text = "We decided to use PostgreSQL for the metadata index because it scales better."
        items = extract_from_text(text)

        assert len(items) >= 1
        decision_items = [i for i in items if i["type"] == "decision"]
        assert len(decision_items) >= 1
        assert any("PostgreSQL" in i["content"] for i in decision_items)

    def test_extract_patterns(self):
        """extract_from_text finds pattern keywords."""
        from memra_local.hooks.extractor import extract_from_text

        text = "Always use atomic writes when writing to flat files to prevent corruption."
        items = extract_from_text(text)

        pattern_items = [i for i in items if i["type"] == "pattern"]
        assert len(pattern_items) >= 1

    def test_extract_facts(self):
        """extract_from_text finds fact/learning patterns."""
        from memra_local.hooks.extractor import extract_from_text

        text = "We learned that the old caching layer drops keys under load."
        items = extract_from_text(text)

        fact_items = [i for i in items if i["type"] == "fact"]
        assert len(fact_items) >= 1
        assert any("caching" in i["content"] or "drops" in i["content"] for i in fact_items)

    def test_extract_dedup(self):
        """Duplicate content within extraction is deduplicated."""
        from memra_local.hooks.extractor import extract_from_messages

        messages = [
            "We decided to use atomic writes for file safety.",
            "We decided to use atomic writes for file safety.",  # exact dup
        ]
        items = extract_from_messages(messages)

        # Should have exactly 1, not 2
        contents = [i["content"] for i in items]
        assert len(contents) == len(set(c[:100].lower() for c in contents))

    def test_extract_length_bounds(self):
        """Matches shorter than 10 chars or longer than 500 chars are skipped."""
        from memra_local.hooks.extractor import extract_from_text

        # Short match
        short_text = "We decided to x."
        items_short = extract_from_text(short_text)
        for item in items_short:
            assert len(item["content"]) >= 10

        # Long match (construct artificially long text)
        long_content = "a" * 600
        long_text = f"We decided to {long_content} for no reason."
        items_long = extract_from_text(long_text)
        for item in items_long:
            assert len(item["content"]) <= 500


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Tests for hooks install/uninstall and capture CLI commands."""

    def test_hooks_install_cli(self, tmp_path: Path):
        """hooks install command works via CLI runner."""
        from memra_local.cli import cli

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"

        runner = CliRunner()
        with patch(
            "memra_local.hooks.installer.CLAUDE_SETTINGS", settings_path
        ):
            result = runner.invoke(cli, ["hooks", "install"])

        assert result.exit_code == 0
        assert "installed" in result.output.lower()

    def test_hooks_uninstall_cli(self, tmp_path: Path):
        """hooks uninstall command works via CLI runner."""
        from memra_local.cli import cli

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        settings_path = claude_dir / "settings.json"
        settings_path.write_text("{}", encoding="utf-8")

        runner = CliRunner()
        with patch(
            "memra_local.hooks.installer.CLAUDE_SETTINGS", settings_path
        ):
            result = runner.invoke(cli, ["hooks", "uninstall"])

        assert result.exit_code == 0
        assert "not installed" in result.output.lower()

    def test_capture_cli(self, tmp_path: Path):
        """capture command reads stdin JSON, parses transcript, saves memories."""
        from memra_local.cli import cli

        # Create a transcript with extractable content
        entries = [
            {
                "type": "assistant",
                "message": {
                    "content": "We decided to use PostgreSQL for the metadata index because it handles concurrent queries well."
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": "Always use atomic writes when dealing with flat file storage."
                },
            },
        ]
        transcript_path = tmp_path / "session.jsonl"
        with open(transcript_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # Storage dir for memories
        storage_dir = tmp_path / "memra-storage"
        storage_dir.mkdir()

        stdin_json = json.dumps(
            {
                "session_id": "test-session-123",
                "transcript_path": str(transcript_path),
            }
        )

        runner = CliRunner()
        with patch(
            "memra_local.cli.create_service"
        ) as mock_create:
            from memra_local.services.factory import create_service as _real

            svc = _real(scope="global", storage_dir=storage_dir)
            mock_create.return_value = svc

            result = runner.invoke(cli, ["capture"], input=stdin_json)

        assert result.exit_code == 0

        # Verify memories were saved
        memories, total = svc.list_()
        assert total > 0

        # Verify trust_state=proposed in metadata
        for mem in memories:
            metadata = mem.get("metadata") or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if metadata.get("trust_state"):
                assert metadata["trust_state"] == "proposed"

        # Verify hookSpecificOutput is in stdout (find JSON line in output)
        for line in result.output.strip().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    output_json = json.loads(line)
                    if "hookSpecificOutput" in output_json:
                        break
                except json.JSONDecodeError:
                    continue
        else:
            # If no JSON found, the hook result might be in mixed output
            assert "hookSpecificOutput" in result.output

    def test_capture_no_stdin(self):
        """capture with empty stdin exits gracefully (exit code 0)."""
        from memra_local.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["capture"], input="")

        assert result.exit_code == 0
        # Should still produce JSON output
        output_json = json.loads(result.output.strip())
        assert "hookSpecificOutput" in output_json


# ---------------------------------------------------------------------------
# Trust state filtering tests (LOCAL-05h through LOCAL-05l)
# ---------------------------------------------------------------------------

def _create_test_service(tmp_path: Path):
    """Create a MemoryService with tmp storage for testing."""
    from memra_local.services.factory import create_service

    storage_dir = tmp_path / "memra-trust-test"
    storage_dir.mkdir()
    return create_service(scope="global", storage_dir=storage_dir)


class TestTrustStateFiltering:
    """Tests for trust_state filtering in recall queries."""

    def test_recall_trust_filter(self, tmp_path: Path):
        """LOCAL-05h: Recall with proposed/rejected/verified returns only verified."""
        svc = _create_test_service(tmp_path)

        # Add 3 memories with different trust states
        svc.add({
            "content": "Verified memory about PostgreSQL indexing strategies",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"trust_state": "verified"},
        })
        svc.add({
            "content": "Proposed memory about Redis caching patterns for testing",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"trust_state": "proposed"},
        })
        svc.add({
            "content": "Rejected memory about SQLite being bad for production use",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"trust_state": "rejected"},
        })

        results, meta = svc.recall(
            query="database indexing",
            namespace="test-ns",
            tenant_id="local",
        )

        # Only verified should appear
        contents = [r["content"] for r in results]
        assert any("PostgreSQL" in c for c in contents), "Verified memory should appear"
        assert not any("Redis caching" in c for c in contents), "Proposed memory should be excluded"
        assert not any("SQLite being bad" in c for c in contents), "Rejected memory should be excluded"

    def test_recall_backwards_compat(self, tmp_path: Path):
        """LOCAL-05i: Memories with NULL metadata or missing trust_state still appear."""
        svc = _create_test_service(tmp_path)

        # Memory with no metadata at all
        svc.add({
            "content": "Legacy memory with no metadata field at all for backwards compat",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
        })
        # Memory with metadata but no trust_state
        svc.add({
            "content": "Memory with metadata but no trust state key inside it",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"source": "manual"},
        })

        results, meta = svc.recall(
            query="memory backwards",
            namespace="test-ns",
            tenant_id="local",
        )

        assert len(results) == 2, "Both memories without trust_state should appear"

    def test_recall_include_proposed(self, tmp_path: Path):
        """LOCAL-05j: When include_proposed=True, proposed memories also returned."""
        svc = _create_test_service(tmp_path)

        svc.add({
            "content": "Proposed memory about async embedding pipeline design",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"trust_state": "proposed"},
        })

        # Default: proposed excluded
        results_default, _ = svc.recall(
            query="embedding pipeline",
            namespace="test-ns",
            tenant_id="local",
        )
        assert len(results_default) == 0, "Proposed should be excluded by default"

        # With include_proposed: proposed included
        results_incl, _ = svc.recall(
            query="embedding pipeline",
            namespace="test-ns",
            tenant_id="local",
            include_proposed=True,
        )
        assert len(results_incl) == 1, "Proposed should be included when flag is set"

    def test_fts_trust_filter(self, tmp_path: Path):
        """LOCAL-05k: FTS search also excludes proposed/rejected memories."""
        svc = _create_test_service(tmp_path)

        svc.add({
            "content": "Verified fact about atomic write operations for file safety",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 5,
            "metadata": {"trust_state": "verified"},
        })
        svc.add({
            "content": "Proposed fact about atomic write patterns in concurrent systems",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 5,
            "metadata": {"trust_state": "proposed"},
        })

        results = svc.search(
            query="atomic write",
            namespace="test-ns",
            tenant_id="local",
        )

        assert len(results) == 1, "Only verified should appear in FTS"
        assert "file safety" in results[0]["content"]

    def test_semantic_trust_filter(self, tmp_path: Path):
        """LOCAL-05l: Semantic recall (get_candidates_with_embeddings) excludes proposed/rejected."""
        svc = _create_test_service(tmp_path)

        # Add memories directly to index with embeddings to test semantic path
        import hashlib
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Only test if embedding service exists
        if svc.embedding_service is None:
            pytest.skip("No embedding service available for semantic test")

        svc.add({
            "content": "Verified semantic memory about cosine similarity scoring",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"trust_state": "verified"},
        })
        svc.add({
            "content": "Proposed semantic memory about vector distance calculations",
            "project_id": "test-ns",
            "type": "fact",
            "importance": 8,
            "metadata": {"trust_state": "proposed"},
        })

        # Get candidates directly from index
        candidates = svc.index.get_candidates_with_embeddings(
            namespace="test-ns",
            tenant_id="local",
        )

        # Only verified should have candidates
        for c in candidates:
            meta = c.get("metadata")
            if meta:
                import json as _json
                if isinstance(meta, str):
                    meta = _json.loads(meta)
                trust = meta.get("trust_state")
                assert trust != "proposed", "Proposed should be excluded from candidates"
                assert trust != "rejected", "Rejected should be excluded from candidates"


# ---------------------------------------------------------------------------
# Review CLI tests (LOCAL-05h through LOCAL-05l review workflow)
# ---------------------------------------------------------------------------

class TestReviewCLI:
    """Tests for the `memra review` CLI command."""

    def _setup_proposed(self, tmp_path: Path):
        """Create a service with proposed memories for testing."""
        from memra_local.services.factory import create_service

        storage_dir = tmp_path / "review-test"
        storage_dir.mkdir()
        svc = create_service(scope="global", storage_dir=storage_dir)

        # Add some proposed memories
        svc.add({
            "content": "Proposed memory about PostgreSQL indexing strategies for review test",
            "project_id": "auto-capture",
            "type": "fact",
            "importance": 5,
            "metadata": {"trust_state": "proposed", "session_id": "sess-001"},
        })
        svc.add({
            "content": "Proposed memory about Redis cache invalidation patterns for review",
            "project_id": "auto-capture",
            "type": "pattern",
            "importance": 5,
            "metadata": {"trust_state": "proposed", "session_id": "sess-002"},
        })
        return svc, storage_dir

    def test_review_list(self, tmp_path: Path):
        """review command lists proposed memories."""
        from memra_local.cli import cli

        svc, storage_dir = self._setup_proposed(tmp_path)

        runner = CliRunner()
        with patch("memra_local.cli.create_service", return_value=svc):
            # Use batch verify to avoid interactive prompts, but check output
            result = runner.invoke(cli, ["review", "--batch", "verify"])

        assert result.exit_code == 0
        assert "2 verified" in result.output

    def test_review_verify(self, tmp_path: Path):
        """--batch verify promotes all proposed to verified."""
        from memra_local.cli import cli

        svc, storage_dir = self._setup_proposed(tmp_path)

        runner = CliRunner()
        with patch("memra_local.cli.create_service", return_value=svc):
            result = runner.invoke(cli, ["review", "--batch", "verify"])

        assert result.exit_code == 0
        assert "2 verified" in result.output

        # Verify trust_state changed
        memories, _ = svc.list_()
        for mem in memories:
            metadata = mem.get("metadata") or {}
            if isinstance(metadata, str):
                import json as _json
                metadata = _json.loads(metadata)
            if metadata.get("trust_state"):
                assert metadata["trust_state"] == "verified"

    def test_review_reject(self, tmp_path: Path):
        """--batch reject marks all proposed as rejected."""
        from memra_local.cli import cli

        svc, storage_dir = self._setup_proposed(tmp_path)

        runner = CliRunner()
        with patch("memra_local.cli.create_service", return_value=svc):
            result = runner.invoke(cli, ["review", "--batch", "reject"])

        assert result.exit_code == 0
        assert "2 rejected" in result.output

        # Verify trust_state changed
        memories, _ = svc.list_()
        for mem in memories:
            metadata = mem.get("metadata") or {}
            if isinstance(metadata, str):
                import json as _json
                metadata = _json.loads(metadata)
            if metadata.get("trust_state"):
                assert metadata["trust_state"] == "rejected"

    def test_review_empty(self, tmp_path: Path):
        """No proposed memories shows appropriate message."""
        from memra_local.cli import cli
        from memra_local.services.factory import create_service

        storage_dir = tmp_path / "empty-review"
        storage_dir.mkdir()
        svc = create_service(scope="global", storage_dir=storage_dir)

        runner = CliRunner()
        with patch("memra_local.cli.create_service", return_value=svc):
            result = runner.invoke(cli, ["review"])

        assert result.exit_code == 0
        assert "No proposed memories" in result.output

    def test_review_verified_in_recall(self, tmp_path: Path):
        """After verifying, memory appears in recall results."""
        from memra_local.cli import cli

        svc, storage_dir = self._setup_proposed(tmp_path)

        # Before review: proposed memories excluded from recall
        results_before, _ = svc.recall(
            query="PostgreSQL indexing",
            namespace="auto-capture",
            tenant_id="local",
        )
        assert len(results_before) == 0, "Proposed should not appear in recall"

        # Verify via batch
        runner = CliRunner()
        with patch("memra_local.cli.create_service", return_value=svc):
            result = runner.invoke(cli, ["review", "--batch", "verify"])
        assert result.exit_code == 0

        # After review: verified memories appear in recall
        results_after, _ = svc.recall(
            query="PostgreSQL indexing",
            namespace="auto-capture",
            tenant_id="local",
        )
        assert len(results_after) > 0, "Verified memory should appear in recall"
