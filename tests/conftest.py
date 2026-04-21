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
