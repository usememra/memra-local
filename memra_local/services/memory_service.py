"""Business logic orchestrating dual-write storage (flat file + SQLite index)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
from ulid import ULID

from memra_local.exceptions import ConcurrentModificationError
from memra_local.storage.flat_file import FlatFileStore
from memra_local.storage.sqlite_index import SQLiteIndex

logger = logging.getLogger("memra.memory")

if TYPE_CHECKING:
    from memra_local.services.embedding_service import EmbeddingService
    from memra_local.services.sync_service import SyncService


class MemoryService:
    """Orchestrates memory operations across flat-file store and SQLite index."""

    def __init__(
        self,
        store: FlatFileStore,
        index: SQLiteIndex,
        embedding_service: EmbeddingService | None = None,
        sync_service: SyncService | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.embedding_service = embedding_service
        self.sync_service = sync_service

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------
    def add(self, request: dict) -> tuple[dict, bool]:
        """Add a new memory with dedup check.

        Args:
            request: Dict with content, tenant_id, project_id, type,
                     importance, tags, source, metadata.

        Returns:
            (memory_data, is_duplicate) -- is_duplicate=True means existing returned.
        """
        content: str = request["content"]
        tenant_id: str = request.get("tenant_id", "local")
        project_id: str = request.get("project_id", "default")

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Dedup: check if this content already exists for this namespace+tenant
        existing = self.index.find_by_hash(project_id, tenant_id, content_hash)
        if existing:
            # Read flat file to get full content
            try:
                file_data = self.store.read(existing["storage_path"])
                return file_data, True
            except FileNotFoundError:
                # Index has a stale entry -- delete it and proceed to create
                self.index.delete_by_id(existing["id"])

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        memory_id = f"mem_{ULID()}"

        memory_data = {
            "id": memory_id,
            "content": content,
            "tenant_id": tenant_id,
            "project_id": project_id,
            "type": request.get("type", "fact"),
            "importance": request.get("importance", 5),
            "tags": request.get("tags") or [],
            "source": request.get("source"),
            "metadata": request.get("metadata"),
            "embedding_status": "none",
            "expires_at": None,
            "confidence": 1.0,
            "staleness_score": 0,
            "last_accessed_at": None,
            "status": "active",
            "superseded_by": None,
            "ttl_days": None,
            "soft_ttl_days": None,
            "created_at": now,
            "updated_at": now,
        }

        # Generate embedding if service available
        embedding_blob: bytes | None = None
        if self.embedding_service is not None:
            embedding = self.embedding_service.encode(content)
            embedding_blob = self.embedding_service.serialize(embedding)
            memory_data["embedding_status"] = "complete"

        # Dual write: flat file first, then index
        storage_path = self.store.write(project_id, memory_id, memory_data)

        try:
            self.index.insert_with_embedding(
                memory_id=memory_id,
                namespace=project_id,
                tenant_id=tenant_id,
                type_=memory_data["type"],
                importance=memory_data["importance"],
                tags=memory_data["tags"],
                content_hash=content_hash,
                storage_path=storage_path,
                content=content,
                source=memory_data["source"],
                metadata=memory_data["metadata"],
                created_at=now,
                updated_at=now,
                embedding=embedding_blob,
            )
        except Exception:
            # Rollback flat file on index failure
            self.store.delete(storage_path)
            raise

        # Record sync event for synced namespaces
        if self.sync_service:
            self.sync_service.record_event(
                "memory_created", project_id, memory_id, memory_data,
            )

        return memory_data, False

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------
    def get(self, memory_id: str) -> dict | None:
        """Get a single memory by ID, reading content from flat file."""
        row = self.index.get_by_id(memory_id)
        if row is None:
            return None

        try:
            file_data = self.store.read(row["storage_path"])
            return file_data
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    def list_(
        self,
        namespace: str | None = None,
        tenant_id: str | None = None,
        type_: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List memories with optional filters.

        Returns:
            (memories, total_count)
        """
        rows, total = self.index.list_memories(
            namespace=namespace,
            tenant_id=tenant_id,
            type_=type_,
            limit=limit,
            offset=offset,
        )

        memories = []
        for row in rows:
            memory = self._row_to_list_item(row)
            memories.append(memory)

        return memories, total

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        memory_id: str,
        updates: dict,
        expected_revision: int | None = None,
    ) -> dict | None:
        """Update a memory's content and/or metadata.

        Args:
            memory_id: The memory to update.
            updates: Dict of fields to update.
            expected_revision: If provided, use optimistic locking. Raises
                ConcurrentModificationError on version mismatch.
        """
        existing = self.index.get_by_id(memory_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Read current flat-file data
        try:
            file_data = self.store.read(existing["storage_path"])
        except FileNotFoundError:
            return None

        # Build index updates
        index_updates: dict = {"updated_at": now}

        # Apply updates to file data
        if "content" in updates:
            file_data["content"] = updates["content"]
            content_hash = hashlib.sha256(
                updates["content"].encode("utf-8")
            ).hexdigest()
            index_updates["content_hash"] = content_hash
        if "importance" in updates:
            file_data["importance"] = updates["importance"]
            index_updates["importance"] = updates["importance"]
        if "tags" in updates:
            file_data["tags"] = updates["tags"]
            index_updates["tags"] = updates["tags"]
        if "metadata" in updates:
            file_data["metadata"] = updates["metadata"]
            index_updates["metadata"] = updates["metadata"]
        if "type" in updates:
            file_data["type"] = updates["type"]
            index_updates["type"] = updates["type"]

        file_data["updated_at"] = now

        # Write flat file atomically
        namespace = existing["namespace"]
        self.store.write(namespace, memory_id, file_data)

        # Update index (with or without optimistic locking)
        if expected_revision is not None:
            rowcount = self.index.update_by_id_locked(
                memory_id, index_updates, expected_revision
            )
            if rowcount == 0:
                raise ConcurrentModificationError(memory_id)
        else:
            self.index.update_by_id(memory_id, index_updates)

        # Update FTS content if content changed
        if "content" in updates:
            self.index._c.execute(
                "DELETE FROM memories_fts WHERE id = ?", (memory_id,)
            )
            self.index._c.execute(
                "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
                (memory_id, updates["content"]),
            )
            self.index._c.commit()

            # Re-generate embedding for updated content
            if self.embedding_service is not None:
                embedding = self.embedding_service.encode(updates["content"])
                embedding_blob = self.embedding_service.serialize(embedding)
                self.index.update_embedding(memory_id, embedding_blob)
                file_data["embedding_status"] = "complete"

        # Record sync event for synced namespaces
        if self.sync_service:
            namespace = existing["namespace"]
            base_rev = existing.get("revision", 1)
            self.sync_service.record_event(
                "memory_updated", namespace, memory_id, file_data,
                base_revision=base_rev,
            )

        return file_data

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete(self, memory_id: str) -> bool:
        """Delete a memory from both index and flat file."""
        row = self.index.get_by_id(memory_id)
        if row is None:
            return False

        storage_path = row["storage_path"]
        namespace = row["namespace"]

        # Delete from index first (source of truth for existence)
        self.index.delete_by_id(memory_id)

        # Delete flat file
        self.store.delete(storage_path)

        # Record sync event for synced namespaces
        if self.sync_service:
            self.sync_service.record_event(
                "memory_deleted", namespace, memory_id, {"id": memory_id},
            )

        return True

    # ------------------------------------------------------------------
    # Supersede
    # ------------------------------------------------------------------
    def supersede(self, old_id: str, new_content: str, **kwargs) -> tuple[dict, dict]:
        """Supersede a memory: create replacement, mark old as superseded.

        Args:
            old_id: ID of the memory to supersede.
            new_content: Content for the replacement memory.
            **kwargs: Optional overrides for type, importance, tags, metadata.

        Returns:
            (new_memory_data, old_memory_data) tuple.

        Raises:
            ValueError: If old memory not found or already superseded.
            ConcurrentModificationError: If version conflict detected.
        """
        old_row = self.index.get_by_id(old_id)
        if old_row is None:
            raise ValueError(f"Memory not found: {old_id}")

        if old_row.get("status") == "superseded":
            raise ValueError(f"Memory already superseded: {old_id}")

        expected_revision = old_row.get("revision") or 1

        # Build request for new memory, inheriting from old
        new_request = {
            "content": new_content,
            "tenant_id": old_row.get("tenant_id", "local"),
            "project_id": old_row.get("namespace", "default"),
            "type": kwargs.get("type", old_row.get("type", "fact")),
            "importance": kwargs.get("importance", old_row.get("importance", 5)),
            "tags": kwargs.get("tags", json.loads(old_row["tags"]) if isinstance(old_row.get("tags"), str) else old_row.get("tags", [])),
            "source": old_row.get("source"),
            "metadata": kwargs.get("metadata", json.loads(old_row["metadata"]) if isinstance(old_row.get("metadata"), str) and old_row.get("metadata") else old_row.get("metadata")),
        }

        # Create new memory
        new_data, _ = self.add(new_request)
        new_id = new_data["id"]

        # Mark old as superseded with optimistic locking
        rowcount = self.index.supersede_by_id(old_id, new_id, expected_revision)
        if rowcount == 0:
            # Version conflict -- rollback new memory.
            # Read the flat-file path BEFORE deleting the index row, so we can
            # still clean up the orphaned flat file.
            try:
                new_row = self.index.get_by_id(new_id)
                self.index.delete_by_id(new_id)
                if new_row:
                    self.store.delete(new_row["storage_path"])
            except Exception as exc:
                logger.warning(
                    "supersede rollback failed for new_id=%s: %s (orphaned flat file may remain)",
                    new_id, exc,
                )
            raise ConcurrentModificationError(old_id)

        # Update old flat file to reflect superseded state
        try:
            old_file_data = self.store.read(old_row["storage_path"])
            old_file_data["superseded_by"] = new_id
            old_file_data["status"] = "superseded"
            self.store.write(old_row["namespace"], old_id, old_file_data)
        except FileNotFoundError:
            old_file_data = {"id": old_id, "status": "superseded", "superseded_by": new_id}

        # Record sync event
        if self.sync_service:
            self.sync_service.record_event(
                "memory_superseded",
                old_row.get("namespace", "default"),
                old_id,
                {"old_id": old_id, "new_id": new_id},
            )

        return new_data, old_file_data

    # ------------------------------------------------------------------
    # Chain Walking
    # ------------------------------------------------------------------
    def get_chain(self, memory_id: str) -> list[dict]:
        """Walk supersession chain and return full history.

        Args:
            memory_id: Any memory ID in the chain.

        Returns:
            List of memory data dicts, ordered oldest to newest.
            Empty list if memory_id not found.
        """
        rows = self.index.get_chain_rows(memory_id)
        if not rows:
            return []

        chain: list[dict] = []
        for row in rows:
            try:
                file_data = self.store.read(row["storage_path"])
                chain.append(file_data)
            except FileNotFoundError:
                continue

        return chain

    # ------------------------------------------------------------------
    # Search (FTS5)
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        namespace: str,
        tenant_id: str,
        type_: str | None = None,
        importance_min: int | None = None,
        limit: int = 20,
        offset: int = 0,
        include_proposed: bool = False,
    ) -> list[dict]:
        """Search memories using FTS5 full-text search, enriching with flat file content."""
        rows = self.index.search_fts(
            query=query,
            namespace=namespace,
            tenant_id=tenant_id,
            type_=type_,
            importance_min=importance_min,
            limit=limit,
            offset=offset,
            include_proposed=include_proposed,
        )

        results = []
        for row in rows:
            try:
                file_data = self.store.read(row["storage_path"])
                file_data["score"] = abs(row.get("rank", 0))
                results.append(file_data)
            except FileNotFoundError:
                continue

        return results

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    def recall(
        self,
        query: str,
        namespace: str,
        tenant_id: str,
        type_: str | None = None,
        importance_min: int | None = None,
        limit: int = 10,
        include_proposed: bool = False,
    ) -> tuple[list[dict], dict]:
        """Recall memories using semantic search with FTS5 fallback.

        Strategy:
        1. If embedding_service available, try semantic (cosine similarity) search
        2. Fall back to FTS5 text-match if no embeddings or no embedding_service

        Args:
            include_proposed: If True, include trust_state='proposed' memories in results.

        Returns:
            (results, meta) where meta includes scoring info.
        """
        # Try semantic search first
        if self.embedding_service is not None:
            candidates = self.index.get_candidates_with_embeddings(
                namespace=namespace,
                tenant_id=tenant_id,
                type_=type_,
                importance_min=importance_min,
                include_proposed=include_proposed,
            )
            if candidates:
                return self._recall_semantic(query, candidates, limit)

        # Fallback to FTS5 text-match
        return self._recall_fts(
            query, namespace, tenant_id, type_, importance_min, limit,
            include_proposed=include_proposed,
        )

    def _recall_semantic(
        self,
        query: str,
        candidates: list[dict],
        limit: int,
    ) -> tuple[list[dict], dict]:
        """Rank candidates by cosine similarity + importance scoring."""
        assert self.embedding_service is not None

        query_embedding = self.embedding_service.encode(query)

        # Build embeddings matrix from candidates
        embeddings = []
        valid_candidates = []
        for c in candidates:
            if c.get("embedding") is not None:
                emb = self.embedding_service.deserialize(c["embedding"])
                embeddings.append(emb)
                valid_candidates.append(c)

        if not valid_candidates:
            return [], {
                "total_candidates": 0,
                "returned": 0,
                "scoring": "cosine_similarity",
                "query_cached": False,
                "response_cached": False,
            }

        embeddings_matrix = np.stack(embeddings)
        similarities = self.embedding_service.cosine_similarity(query_embedding, embeddings_matrix)

        # Score: 0.7 * similarity + 0.3 * (importance / 10)
        scored = []
        for i, candidate in enumerate(valid_candidates):
            similarity = float(similarities[i])
            importance = candidate.get("importance", 5)
            score = 0.7 * similarity + 0.3 * (importance / 10)
            scored.append((candidate, round(score, 4)))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:limit]

        results = []
        for candidate, score in scored:
            try:
                file_data = self.store.read(candidate["storage_path"])
                recall_item = {
                    "id": file_data["id"],
                    "content": file_data["content"],
                    "score": score,
                    "type": file_data["type"],
                    "importance": file_data["importance"],
                    "tags": file_data.get("tags", []),
                    "created_at": file_data["created_at"],
                }
                results.append(recall_item)
            except FileNotFoundError:
                continue

        meta = {
            "total_candidates": len(valid_candidates),
            "returned": len(results),
            "scoring": "cosine_similarity",
            "query_cached": False,
            "response_cached": False,
        }

        return results, meta

    def _recall_fts(
        self,
        query: str,
        namespace: str,
        tenant_id: str,
        type_: str | None,
        importance_min: int | None,
        limit: int,
        include_proposed: bool = False,
    ) -> tuple[list[dict], dict]:
        """FTS5 text-match fallback for recall."""
        rows = self.index.search_fts(
            query=query,
            namespace=namespace,
            tenant_id=tenant_id,
            type_=type_,
            importance_min=importance_min,
            limit=limit,
            offset=0,
            include_proposed=include_proposed,
        )

        results = []
        for row in rows:
            try:
                file_data = self.store.read(row["storage_path"])
                recall_item = {
                    "id": file_data["id"],
                    "content": file_data["content"],
                    "score": round(abs(row.get("rank", 0)), 4),
                    "type": file_data["type"],
                    "importance": file_data["importance"],
                    "tags": file_data.get("tags", []),
                    "created_at": file_data["created_at"],
                }
                results.append(recall_item)
            except FileNotFoundError:
                continue

        meta = {
            "total_candidates": len(rows),
            "returned": len(results),
            "scoring": "text_match",
            "query_cached": False,
            "response_cached": False,
        }

        return results, meta

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def bootstrap(
        self,
        namespace: str,
        tenant_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return highest-importance memories for agent bootstrap context.

        When sync is enabled for ``namespace`` (77-04), also fetches
        cloud-promoted memories via ``sync_service.cloud_bootstrap`` and merges
        them with local results. Cloud memories win on id collision because
        the cloud is the source of truth for shared/promoted memories. Cloud
        failures degrade silently to local-only (warning logged).
        """
        # Query index ordered by importance DESC, then created_at DESC
        rows, _ = self.index.list_memories(
            namespace=namespace,
            tenant_id=tenant_id,
            limit=limit,
            offset=0,
        )

        # Sort by importance DESC (list_memories sorts by created_at DESC)
        rows.sort(key=lambda r: r.get("importance", 5), reverse=True)

        local_results: list[dict] = []
        for row in rows:
            try:
                file_data = self.store.read(row["storage_path"])
                local_results.append(file_data)
            except FileNotFoundError:
                continue

        # Cloud merge — only if a sync_service is wired in and enabled.
        cloud_results: list[dict] = []
        if self.sync_service is not None and self.sync_service.is_sync_enabled(namespace):
            cloud_results = self.sync_service.cloud_bootstrap(
                namespace=namespace,
                tenant_id=tenant_id,
                limit=limit,
            )

        if not cloud_results:
            return local_results

        # Merge: dedupe by id, cloud wins.
        cloud_ids = {m.get("id") for m in cloud_results if m.get("id")}
        merged: list[dict] = list(cloud_results)
        for mem in local_results:
            if mem.get("id") not in cloud_ids:
                merged.append(mem)
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_list_item(row: dict) -> dict:
        """Convert an index row to a MemoryListItem-compatible dict."""
        tags = row.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)

        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return {
            "id": row["id"],
            "tenant_id": row.get("tenant_id", "local"),
            "project_id": row.get("namespace", "default"),
            "type": row.get("type", "fact"),
            "importance": row.get("importance", 5),
            "tags": tags,
            "source": row.get("source"),
            "metadata": metadata,
            "embedding_status": "none",
            "expires_at": row.get("expires_at"),
            "confidence": row.get("confidence", 1.0),
            "staleness_score": 0,
            "last_accessed_at": None,
            "status": row.get("status", "active"),
            "superseded_by": None,
            "ttl_days": None,
            "soft_ttl_days": None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
