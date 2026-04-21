"""Performance regression guards for supersession chain walking.

Phase 68-05: duplicate get_chain() at line 712 did an O(n^2) backward walk
reading every superseded row from the DB on each step. With 10K+ memories
this pins the python process for minutes. The kept line-381 version routes
through ``SQLiteIndex.get_chain_rows`` — rewritten here as a single recursive
CTE over an indexed ``superseded_by`` column, giving linear-time traversal.

Two layers are covered:

* ``test_get_chain_rows_perf_10k_depth`` — SQL-only micro-benchmark. This is
  SC-5's real target: the index walk itself must complete in < 100ms at 10K
  depth. Any regression of the CTE or the ``superseded_by`` index fails here.

* ``test_get_chain_perf_10k_depth`` — full service call including flat-file
  reads. File I/O dominates this path (N sequential YAML reads), so the
  threshold reflects that reality. Eliminating file I/O from chain walking
  is a separate concern tracked for a follow-up plan.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from memra_local.services.memory_service import MemoryService
from memra_local.storage.flat_file import FlatFileStore
from memra_local.storage.sqlite_index import SQLiteIndex


@pytest.fixture
def svc():
    tmp = Path(tempfile.mkdtemp(prefix="memra_chain_perf_"))
    store = FlatFileStore(tmp)
    index = SQLiteIndex(tmp / "index.db")
    index.initialize()
    service = MemoryService(store, index)
    yield service
    index.close()
    shutil.rmtree(tmp, ignore_errors=True)


def _build_10k_chain(svc: MemoryService) -> str:
    first, _ = svc.add(
        {
            "content": "v0",
            "tenant_id": "perf",
            "project_id": "default",
            "type": "fact",
        }
    )
    prev_id = first["id"]
    for i in range(1, 10_000):
        new, _ = svc.supersede(prev_id, f"v{i}")
        prev_id = new["id"]
    return prev_id


@pytest.mark.slow
def test_get_chain_rows_perf_10k_depth(svc: MemoryService):
    """SC-5: SQL-layer chain walk over 10K depth must complete in < 100ms.

    This is the real performance target. The CTE + ``idx_memories_superseded_by``
    index make the walk O(N); without either, the query degrades to O(N^2).
    """
    tip_id = _build_10k_chain(svc)

    start = time.perf_counter()
    rows = svc.index.get_chain_rows(tip_id)
    elapsed = time.perf_counter() - start

    assert len(rows) == 10_000
    assert elapsed < 0.1, (
        f"get_chain_rows took {elapsed * 1000:.1f}ms — expected <100ms"
    )


@pytest.mark.slow
def test_get_chain_perf_10k_depth(svc: MemoryService):
    """Service-level get_chain() must complete in a reasonable time at 10K depth.

    Threshold is 10s rather than 100ms because ``MemoryService.get_chain()``
    performs N sequential flat-file reads (one YAML parse per row) after the
    indexed SQL walk. Observed on dev hardware: ~5s at 10K depth. The fix in
    this plan removed the O(N^2) DB walk (which took minutes); eliminating
    file I/O from chain reconstruction is tracked as a follow-up concern.

    This guard exists to catch accidental O(N^2) reintroduction at the
    service level — not to assert the 100ms SC-5 goal (that belongs to
    ``test_get_chain_rows_perf_10k_depth``).
    """
    tip_id = _build_10k_chain(svc)

    start = time.perf_counter()
    chain = svc.get_chain(tip_id)
    elapsed = time.perf_counter() - start

    assert len(chain) == 10_000
    assert elapsed < 10.0, (
        f"get_chain took {elapsed * 1000:.1f}ms — expected <10000ms"
    )
