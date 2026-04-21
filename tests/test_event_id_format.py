"""Lock the event_id format emitted by SyncService.

Canonical form is `evt_{ULID}` with Crockford base32 (uppercase, 26 chars,
excludes I/L/O/U). Cloud validator in api/app/Http/Requests/SyncPushRequest.php
accepts this exact shape — a silent refactor here would reintroduce the
422 storm that plan 77-02 closed.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pytest

from memra_local.services.factory import create_service

EVENT_ID_RE = re.compile(r"^evt_[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture
def tmp_storage():
    d = Path(tempfile.mkdtemp(prefix="memra_evtid_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_recorded_event_id_matches_canonical_format(tmp_storage):
    svc = create_service(scope="global", storage_dir=tmp_storage)
    sync = svc.sync_service
    sync.enable("ns-test", api_key="memra_live_test", api_url="http://localhost")

    sync.record_event(
        event_type="memory_created",
        namespace="ns-test",
        memory_id="mem_1",
        payload={"content": "hello"},
    )

    row = sync._conn.execute(
        "SELECT id FROM sync_events WHERE namespace = ?", ("ns-test",)
    ).fetchone()
    assert row is not None, "record_event did not persist an event"
    event_id = row["id"]

    assert EVENT_ID_RE.match(event_id), (
        f"event_id {event_id!r} does not match canonical evt_{{ULID}} shape. "
        "Cloud SyncPushRequest will 422 — see plan 77-02."
    )
