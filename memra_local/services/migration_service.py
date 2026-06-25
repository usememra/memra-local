"""Migration service for uploading local memories to cloud via batch endpoint."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from memra_local.models import MigrateResult

if TYPE_CHECKING:
    from memra_local.services.pii_client import PiiClient
    from memra_local.storage.flat_file import FlatFileStore
    from memra_local.storage.sqlite_index import SQLiteIndex

logger = logging.getLogger("memra.migration")

BATCH_SIZE = 50


class MigrationService:
    """Uploads local memories to cloud account via POST /v1/memories/batch.

    Supports PII masking, dry-run, progress callbacks, and partial failure resume.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        pii_client: PiiClient | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._pii_client = pii_client

    def migrate(
        self,
        index: SQLiteIndex,
        store: FlatFileStore,
        project_id: str,
        dry_run: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MigrateResult:
        """Migrate all local memories to cloud via batch endpoint.

        Args:
            index: SQLite index to read memory metadata from.
            store: Flat file store to read memory content from.
            project_id: Target cloud project ID (proj_...) every memory is
                migrated into. The cloud batch endpoint requires an existing
                project; a local namespace is not a valid cloud project ID, so
                the caller must supply a real one. Each memory's local namespace
                is preserved as its cloud ``tenant_id``.
            dry_run: If True, count memories without uploading.
            progress_callback: Called with (migrated_so_far, total) after each batch.

        Returns:
            MigrateResult with counts of total, migrated, skipped, failed.
        """
        # Read all memories from index
        all_rows = index._c.execute(
            "SELECT * FROM memories_index ORDER BY created_at ASC"
        ).fetchall()
        all_rows = [dict(r) for r in all_rows]

        total = len(all_rows)

        if dry_run:
            return MigrateResult(
                total=total, migrated=0, skipped=0, failed=0, dry_run=True,
            )

        # Build memory dicts from index rows + flat file content
        memories = []
        for row in all_rows:
            try:
                file_data = store.read(row["storage_path"])
            except FileNotFoundError:
                logger.warning("File not found for %s, skipping", row["id"])
                continue

            tags = row.get("tags", "[]")
            if isinstance(tags, str):
                tags = json.loads(tags)

            metadata = row.get("metadata")
            if isinstance(metadata, str) and metadata:
                metadata = json.loads(metadata)

            memory = {
                "content": file_data.get("content", ""),
                "type": row.get("type", "fact"),
                "importance": row.get("importance", 5),
                "tags": tags,
                "source": row.get("source"),
                "metadata": metadata,
                "tenant_id": row.get("namespace", "default"),
                "project_id": project_id,
            }
            memories.append(memory)

        total = len(memories)
        migrated = 0
        skipped = 0
        failed = 0
        errors: list[str] = []

        # Process in batches
        batches = _chunk(memories, BATCH_SIZE)

        for batch in batches:
            # PII masking if client provided
            if self._pii_client is not None:
                contents = [m["content"] for m in batch]
                masked = self._pii_client.mask_batch(contents)
                if masked is None:
                    # Fail-closed: do not send unmasked data
                    failed += len(batch)
                    errors.append(
                        f"PII masking failed for batch of {len(batch)} memories"
                    )
                    continue
                # Apply masked content back
                for i, m in enumerate(batch):
                    m["content"] = masked[i]

            # Upload batch
            try:
                response = httpx.post(
                    f"{self._api_url}/memories/batch",
                    json={"memories": batch},
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                migrated += data.get("created", 0)
                skipped += data.get("duplicates", 0)
                # 207 Multi-Status: raise_for_status() does not raise on 2xx, so
                # per-item failures (e.g. "No project with ID X exists") would be
                # silently dropped, reproducing the "0 migrated, no error" pain.
                # Count them and surface the first message.
                batch_errors = data.get("errors", 0)
                if batch_errors:
                    failed += batch_errors
                    first = next(
                        (r.get("error", {}).get("message")
                         for r in data.get("results", [])
                         if r.get("status") not in ("created", "duplicate")),
                        None,
                    )
                    if first:
                        errors.append(first)
            except Exception as exc:
                failed += len(batch)
                errors.append(f"Batch upload failed: {exc}")
                continue

            if progress_callback is not None:
                progress_callback(migrated, total)

        return MigrateResult(
            total=total,
            migrated=migrated,
            skipped=skipped,
            failed=failed,
            errors=errors,
            dry_run=False,
        )


def _chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    return [items[i : i + size] for i in range(0, len(items), size)]
