"""Verify check_account_tier never calls the phantom /account endpoint.

The cloud API exposes no `/account` route. `check_account_tier` previously
issued `GET {api_url}/account`, which always 404'd. It now reads tier from
`GET /usage` (the real route), staying fail-closed on any error. This file
guards against the /account regression; the positive /usage behaviour is
covered in test_sync.py::TestCheckAccountTier.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import memra_local.services.sync_service as sync_module
from memra_local.services.factory import create_service


SYNC_SERVICE_PATH = Path(sync_module.__file__)


@pytest.fixture
def sync(tmp_path):
    svc = create_service(scope="global", storage_dir=tmp_path)
    svc.sync_service.enable("ns", api_key="memra_live_test")
    return svc.sync_service


class TestNoAccountCall:
    def test_source_has_no_account_reference(self):
        """sync_service.py must not reference the phantom /account endpoint."""
        source = SYNC_SERVICE_PATH.read_text()
        assert "/account" not in source, (
            "sync_service.py still references /account — the cloud API has "
            "no such endpoint. Remove the call."
        )

    def test_check_account_tier_queries_usage_not_account(self, sync):
        """The tier lookup must hit /usage, never the phantom /account."""
        captured = {}

        def _capture(url, *args, **kwargs):
            captured["url"] = url
            raise httpx.ConnectError("blocked")  # fail-closed, no real network

        with patch.object(httpx, "get", side_effect=_capture):
            result = sync.check_account_tier("ns")

        assert "/usage" in captured["url"]
        assert "/account" not in captured["url"]
        assert result is None  # fail-closed on the connect error

    def test_check_account_tier_returns_none_when_namespace_missing(self, sync):
        assert sync.check_account_tier("not-enabled") is None
