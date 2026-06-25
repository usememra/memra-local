"""Shared test fixtures for memra-local."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_storage() -> Path:
    """Create a temporary storage directory, clean up after test."""
    d = Path(tempfile.mkdtemp(prefix="memra_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_project_dir() -> Path:
    """Create a temporary project directory with .memra/ subdirectory."""
    d = Path(tempfile.mkdtemp(prefix="memra_project_"))
    (d / ".memra").mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_sync_credentials(tmp_path, monkeypatch):
    """Redirect persisted sync API keys to a per-test path.

    SyncService.enable() persists API keys to ~/.memra/credentials.json so
    push/pull from other processes can authenticate. Tests must never write
    fake keys into the real home directory.
    """
    from memra_local.services.sync_service import SyncService

    monkeypatch.setattr(
        SyncService,
        "_credentials_path",
        lambda self: tmp_path / "credentials.json",
    )
