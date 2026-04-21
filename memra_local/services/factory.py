"""Shared service creation factory for memra-local.

Used by: HTTP API (app.py), MCP server (mcp_server.py), CLI commands.
Ensures all entry points create MemoryService the same way.
"""

from __future__ import annotations

from pathlib import Path

from memra_local.config import resolve_scope, resolve_storage_dir
from memra_local.services.embedding_service import EmbeddingService
from memra_local.services.memory_service import MemoryService
from memra_local.services.sync_service import SyncService
from memra_local.storage.flat_file import FlatFileStore
from memra_local.storage.sqlite_index import SQLiteIndex


def create_service(
    scope: str = "auto",
    storage_dir: Path | None = None,
) -> MemoryService:
    """Create a fully initialized MemoryService.

    Args:
        scope: Storage scope -- "global", "project", or "auto".
        storage_dir: Override storage directory (used by tests).

    Returns:
        MemoryService with store, index, and embedding_service wired up.
    """
    resolved_scope = resolve_scope(scope) if storage_dir is None else scope

    if storage_dir is None:
        storage_dir = resolve_storage_dir(resolved_scope)

    # Ensure storage dir exists
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Create .gitignore for project scope (prevent accidental commits)
    gitignore_path = storage_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text("*\n!.gitignore\n", encoding="utf-8")

    # Initialize storage layers
    store = FlatFileStore(base_dir=storage_dir)
    db_path = storage_dir / "memra.db"
    index = SQLiteIndex(db_path=db_path)
    index.initialize()

    # Initialize embedding service (lazy model loading -- fast startup)
    embedding_service = EmbeddingService()

    # Initialize sync service (lightweight -- no network until push/pull)
    sync_service = SyncService(index=index)

    return MemoryService(store=store, index=index, embedding_service=embedding_service, sync_service=sync_service)
