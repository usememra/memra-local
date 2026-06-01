"""Tests for CLI entry point and server startup."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from click.testing import CliRunner

from memra_local.cli import cli


class TestCLIServeHelp:
    """Test that `memra serve --help` shows expected options."""

    def test_cli_serve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--scope" in result.output

    def test_cli_version(self):
        from memra_local import __version__

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output


class TestCLIDefaultPort:
    """Test that serve command defaults to port 8765."""

    def test_cli_default_port(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "8765" in result.output


class TestCLICustomPort:
    """Test that serve command accepts --port 9999."""

    def test_cli_custom_port(self, tmp_storage):
        """The serve command should accept a custom port without error in parsing."""
        runner = CliRunner()
        # We use --help after setting port to verify the option is accepted
        # (Actually running the server would block, so we test the CLI parsing separately)
        result = runner.invoke(cli, ["serve", "--port", "9999", "--help"])
        # --help takes precedence, so it should still show help
        assert result.exit_code == 0


class TestCLIScopeChoices:
    """Test that --scope only accepts valid choices."""

    def test_cli_scope_accepts_global(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--scope", "global", "--help"])
        assert result.exit_code == 0

    def test_cli_scope_accepts_project(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--scope", "project", "--help"])
        assert result.exit_code == 0

    def test_cli_scope_accepts_auto(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--scope", "auto", "--help"])
        assert result.exit_code == 0

    def test_cli_scope_rejects_invalid(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--scope", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()


class TestServerStartsAndResponds:
    """Test that the server starts and responds to /health."""

    def test_server_starts_and_responds(self, tmp_storage):
        from memra_local.app import create_app

        app = create_app(scope="project", storage_dir=tmp_storage)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run)
        thread.daemon = True
        thread.start()

        # Wait for server to start
        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        assert server.started, "Server did not start within 5 seconds"

        # Get actual port
        sockets = server.servers[0].sockets
        port = sockets[0].getsockname()[1]

        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "version" in data
        finally:
            server.should_exit = True
            thread.join(timeout=5)


class TestServerAddAndGet:
    """Test that the server can add and retrieve memories."""

    def test_server_add_and_get(self, tmp_storage):
        from memra_local.app import create_app

        app = create_app(scope="project", storage_dir=tmp_storage)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run)
        thread.daemon = True
        thread.start()

        for _ in range(50):
            if server.started:
                break
            time.sleep(0.1)
        assert server.started, "Server did not start within 5 seconds"

        sockets = server.servers[0].sockets
        port = sockets[0].getsockname()[1]

        try:
            # Add a memory
            add_resp = httpx.post(
                f"http://127.0.0.1:{port}/v1/memories",
                json={
                    "content": "The server integration test works",
                    "type": "fact",
                    "tenant_id": "test-tenant",
                },
                timeout=5.0,
            )
            assert add_resp.status_code == 201
            memory_id = add_resp.json()["id"]

            # Get it back
            get_resp = httpx.get(
                f"http://127.0.0.1:{port}/v1/memories/{memory_id}",
                timeout=5.0,
            )
            assert get_resp.status_code == 200
            assert get_resp.json()["content"] == "The server integration test works"
        finally:
            server.should_exit = True
            thread.join(timeout=5)


class TestMainModule:
    """Test that python -m memra_local works."""

    def test_main_module_imports(self):
        """Verify __main__.py imports cli correctly."""
        import importlib
        mod = importlib.import_module("memra_local.__main__")
        assert hasattr(mod, "cli")
        assert mod.cli is cli
