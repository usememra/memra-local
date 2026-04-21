"""Sync service for push/pull synchronization with Memra cloud."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from ulid import ULID

from memra_local.models import SyncPullResult, SyncPushResult

if TYPE_CHECKING:
    from memra_local.services.memory_service import MemoryService
    from memra_local.services.pii_client import PiiClient
    from memra_local.storage.sqlite_index import SQLiteIndex

logger = logging.getLogger("memra.sync")


class SyncService:
    """Manages namespace sync: enable, record events, push, pull, conflicts."""

    VALID_PII_MODES = ("local_private", "shared_masked", "shared_raw")

    def __init__(self, index: SQLiteIndex, api_key: str | None = None, pii_client: "PiiClient | None" = None) -> None:
        self._index = index
        self._conn = index._c
        self._api_key = api_key
        self._pii_client = pii_client
        self._tier_cache: dict[str, str] = {}
        # Cache of enabled namespaces for fast is_sync_enabled checks
        self._enabled_cache: set[str] = set()
        self._cache_loaded = False

    def _load_cache(self) -> None:
        """Load enabled namespaces into memory cache."""
        if self._cache_loaded:
            return
        try:
            rows = self._conn.execute("SELECT namespace FROM sync_cursors").fetchall()
            self._enabled_cache = {row["namespace"] for row in rows}
        except Exception:
            self._enabled_cache = set()
        self._cache_loaded = True

    # ------------------------------------------------------------------
    # Enable / Disable
    # ------------------------------------------------------------------

    def enable(
        self,
        namespace: str,
        api_key: str,
        api_url: str = "https://usememra.com/api/v1",
        pii_mode: str = "local_private",
        agent_id: str | None = None,
    ) -> None:
        """Enable sync for a namespace. Stores config in sync_cursors table.

        ``agent_id`` is the cloud API-key id required by
        ``/v1/agents/{agent_id}/bootstrap``. If omitted, cloud bootstrap is
        skipped and ``bootstrap()`` stays local-only (77-04).
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """INSERT OR REPLACE INTO sync_cursors
               (namespace, cloud_api_url, remote_cursor, last_synced_at, enabled_at, pii_mode, agent_id)
               VALUES (?, ?, 0, NULL, ?, ?, ?)""",
            (namespace, api_url, now, pii_mode, agent_id),
        )
        self._conn.commit()
        self._enabled_cache.add(namespace)
        # Store API key per-namespace (in memory for this session)
        if not hasattr(self, "_api_keys"):
            self._api_keys: dict[str, str] = {}
        self._api_keys[namespace] = api_key

    def disable(self, namespace: str) -> bool:
        """Disable sync for a namespace."""
        cursor = self._conn.execute(
            "DELETE FROM sync_cursors WHERE namespace = ?", (namespace,)
        )
        self._conn.commit()
        self._enabled_cache.discard(namespace)
        return cursor.rowcount > 0

    def is_sync_enabled(self, namespace: str) -> bool:
        """Check if namespace has sync enabled. Fast path using cache."""
        self._load_cache()
        return namespace in self._enabled_cache

    # ------------------------------------------------------------------
    # PII mode management
    # ------------------------------------------------------------------

    def set_mode(self, namespace: str, mode: str) -> bool:
        """Set PII sharing mode for a namespace. Returns True if updated."""
        if mode not in self.VALID_PII_MODES:
            return False
        cursor = self._conn.execute(
            "UPDATE sync_cursors SET pii_mode = ? WHERE namespace = ?",
            (mode, namespace),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_mode(self, namespace: str) -> str | None:
        """Get PII sharing mode for a namespace. Returns None if not found."""
        row = self._conn.execute(
            "SELECT pii_mode FROM sync_cursors WHERE namespace = ?", (namespace,)
        ).fetchone()
        return row["pii_mode"] if row else None

    def check_account_tier(self, namespace: str) -> str | None:
        """Return the account tier for a synced namespace, or None if unknown.

        Historical note (Phase 77-03): this used to issue a GET against a phantom
        account endpoint, but no such route exists on the cloud API — the call
        always 404'd. The cloud ``/v1/agents/{id}/bootstrap`` response also does
        not surface tier or quota, so there is no alternate source on v1.0.

        Returning ``None`` is fail-closed: the only caller that gates on the tier
        value (``shared_raw`` push mode) rejects any non-{team,admin} result, so
        an unknown tier still blocks raw-PII uploads. When the cloud exposes tier
        information in a future release, wire it in here.
        """
        # No cloud endpoint exposes tier in v1.0 — always fail-closed.
        # When a tier/quota endpoint lands, populate self._tier_cache here
        # and return the cached value.
        config = self._conn.execute(
            "SELECT 1 FROM sync_cursors WHERE namespace = ?", (namespace,)
        ).fetchone()
        if config is None:
            return None
        return None  # explicit: no source for tier yet

    def _mask_event_payloads(self, events: list[dict], namespace: str) -> list[dict] | None:
        """Mask PII in event payloads for shared_masked mode.

        Returns updated events list, or None if masking failed.
        """
        if self._pii_client is None:
            return None

        # Collect content strings that need masking
        content_indices: list[int] = []
        content_strings: list[str] = []
        for i, event in enumerate(events):
            if event["event_type"] in ("memory_created", "memory_updated"):
                content = event.get("payload", {}).get("content")
                if content:
                    content_indices.append(i)
                    content_strings.append(content)

        if not content_strings:
            return events  # Nothing to mask

        masked = self._pii_client.mask_batch(content_strings)
        if masked is None:
            return None

        # Replace content in event payloads
        for idx, masked_content in zip(content_indices, masked):
            events[idx]["payload"]["content"] = masked_content

        return events

    # ------------------------------------------------------------------
    # Record event
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        namespace: str,
        memory_id: str,
        payload: dict,
        base_revision: int | None = None,
    ) -> None:
        """Record a sync event. No-op if namespace not synced. Never raises."""
        try:
            if not self.is_sync_enabled(namespace):
                return
            event_id = f"evt_{ULID()}"
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self._conn.execute(
                """INSERT INTO sync_events
                   (id, namespace, event_type, memory_id, payload, base_revision, pushed_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, ?)""",
                (event_id, namespace, event_type, memory_id, json.dumps(payload), base_revision, now),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("Failed to record sync event: %s", exc)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(self, namespace: str) -> SyncPushResult:
        """Push unpushed events to cloud. Up to 100 per call.

        Enforces PII modes:
        - local_private: push blocked entirely
        - shared_masked: content masked via PiiClient before push
        - shared_raw: requires team/admin tier
        """
        rows = self._conn.execute(
            """SELECT * FROM sync_events
               WHERE namespace = ? AND pushed_at IS NULL
               ORDER BY created_at ASC LIMIT 100""",
            (namespace,),
        ).fetchall()

        if not rows:
            return SyncPushResult(pushed=0, cursor=None)

        events = []
        event_ids = []
        for row in rows:
            events.append({
                "event_id": row["id"],
                "event_type": row["event_type"],
                "memory_id": row["memory_id"],
                "payload": json.loads(row["payload"]),
                "base_revision": row["base_revision"],
            })
            event_ids.append(row["id"])

        # Get API config
        config = self._conn.execute(
            "SELECT * FROM sync_cursors WHERE namespace = ?", (namespace,)
        ).fetchone()
        if config is None:
            return SyncPushResult(pushed=0, error="Namespace not sync-enabled")

        pii_mode = config["pii_mode"] if "pii_mode" in config.keys() else "shared_masked"

        # Enforce PII modes
        if pii_mode == "local_private":
            return SyncPushResult(pushed=0, error="Namespace is local_private -- sync disabled")

        if pii_mode == "shared_raw":
            tier = self.check_account_tier(namespace)
            if tier not in ("team", "admin"):
                return SyncPushResult(
                    pushed=0,
                    error=f"shared_raw requires Team or Admin tier (current: {tier})",
                )

        if pii_mode == "shared_masked":
            masked_events = self._mask_event_payloads(events, namespace)
            if masked_events is None:
                return SyncPushResult(
                    pushed=0,
                    error="PII masking failed -- push aborted to prevent unmasked data upload",
                )
            events = masked_events

        api_url = config["cloud_api_url"]
        api_key = getattr(self, "_api_keys", {}).get(namespace, self._api_key or "")

        # Send with retries
        last_error = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    f"{api_url}/sync/push",
                    json={"namespace": namespace, "events": events},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                # Mark events as pushed
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                placeholders = ",".join("?" * len(event_ids))
                self._conn.execute(
                    f"UPDATE sync_events SET pushed_at = ? WHERE id IN ({placeholders})",
                    [now] + event_ids,
                )

                # Update cursor
                if data.get("cursor") is not None:
                    self._conn.execute(
                        "UPDATE sync_cursors SET remote_cursor = ?, last_synced_at = ? WHERE namespace = ?",
                        (data["cursor"], now, namespace),
                    )
                self._conn.commit()

                return SyncPushResult(
                    pushed=data.get("pushed", len(events)),
                    conflicts=data.get("conflicts", []),
                    cursor=data.get("cursor"),
                )
            except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s

        return SyncPushResult(pushed=0, error=f"Push failed after 3 attempts: {last_error}")

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    def pull(self, namespace: str, memory_service: MemoryService) -> SyncPullResult:
        """Pull remote events since cursor and apply locally."""
        config = self._conn.execute(
            "SELECT * FROM sync_cursors WHERE namespace = ?", (namespace,)
        ).fetchone()
        if config is None:
            return SyncPullResult(applied=0, cursor=0, has_more=False, error="Namespace not sync-enabled")

        api_url = config["cloud_api_url"]
        cursor = config["remote_cursor"]
        api_key = getattr(self, "_api_keys", {}).get(namespace, self._api_key or "")

        # Fetch with retries
        last_error = None
        for attempt in range(3):
            try:
                response = httpx.get(
                    f"{api_url}/sync/pull",
                    params={"namespace": namespace, "cursor": cursor},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                events = data.get("events", [])
                applied = 0

                for event in events:
                    event_type = event.get("event_type", "")
                    payload = event.get("payload", {})

                    try:
                        if event_type == "memory_created":
                            memory_service.add({
                                "content": payload.get("content", ""),
                                "project_id": payload.get("project_id", namespace),
                                "type": payload.get("type", "fact"),
                                "importance": payload.get("importance", 5),
                                "tags": payload.get("tags", []),
                                "source": payload.get("source"),
                                "metadata": payload.get("metadata"),
                            })
                            applied += 1
                        elif event_type == "memory_updated":
                            memory_id = event.get("memory_id", "")
                            if memory_id:
                                memory_service.update(memory_id, payload)
                                applied += 1
                        elif event_type == "memory_deleted":
                            memory_id = event.get("memory_id", "")
                            if memory_id:
                                memory_service.delete(memory_id)
                                applied += 1
                    except Exception as exc:
                        logger.warning("Failed to apply event %s: %s", event.get("event_id"), exc)

                # Update cursor after successful apply
                new_cursor = data.get("cursor", cursor)
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                self._conn.execute(
                    "UPDATE sync_cursors SET remote_cursor = ?, last_synced_at = ? WHERE namespace = ?",
                    (new_cursor, now, namespace),
                )
                self._conn.commit()

                return SyncPullResult(
                    applied=applied,
                    cursor=new_cursor,
                    has_more=data.get("has_more", False),
                )
            except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = str(exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return SyncPullResult(applied=0, cursor=cursor, has_more=False, error=f"Pull failed after 3 attempts: {last_error}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, namespace: str) -> dict | None:
        """Return sync config for namespace or None."""
        row = self._conn.execute(
            "SELECT * FROM sync_cursors WHERE namespace = ?", (namespace,)
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------------

    def list_conflicts(self, namespace: str | None = None) -> list[dict]:
        """List memories whose ID contains '_conflict_'."""
        if namespace:
            rows = self._conn.execute(
                "SELECT * FROM memories_index WHERE namespace = ? AND id LIKE '%_conflict_%'",
                (namespace,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories_index WHERE id LIKE '%_conflict_%'"
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_conflict(self, conflict_id: str, keep: str, memory_service: MemoryService) -> bool:
        """Resolve conflict. keep='local': delete conflict sibling. keep='remote': replace original with conflict content, delete sibling."""
        if "_conflict_" not in conflict_id:
            return False

        # Extract original ID (everything before _conflict_)
        original_id = conflict_id.split("_conflict_")[0]

        conflict_memory = memory_service.get(conflict_id)
        if conflict_memory is None:
            return False

        if keep == "local":
            # Delete the conflict sibling, keep original
            memory_service.delete(conflict_id)
            return True
        elif keep == "remote":
            # Capture remote content before any destructive op, then:
            # 1. delete conflict sibling (frees the content_hash unique constraint)
            # 2. update original with remote content
            # If update fails after delete, restore the conflict sibling as a fresh
            # memory so the remote content is not lost (cloud-wins semantics).
            remote_content = conflict_memory["content"]
            if memory_service.get(original_id) is None:
                # Original gone — nothing to merge into; leave conflict in place.
                return False
            memory_service.delete(conflict_id)
            try:
                updated = memory_service.update(original_id, {"content": remote_content})
            except Exception:
                updated = None
            if updated is None:
                # Update failed — re-create the conflict memory to avoid data loss.
                try:
                    restore_payload = {
                        "content": remote_content,
                        "tenant_id": conflict_memory.get("tenant_id", "local"),
                        "project_id": conflict_memory.get("namespace", "default"),
                        "type": conflict_memory.get("type", "fact"),
                        "importance": conflict_memory.get("importance", 5),
                    }
                    memory_service.add(restore_payload)
                except Exception as exc:
                    logger.error(
                        "resolve_conflict remote-update failed and restore failed for "
                        "original_id=%s: %s (remote content lost)",
                        original_id, exc,
                    )
                return False
            return True

        return False

    # ------------------------------------------------------------------
    # Cloud bootstrap (77-04)
    # ------------------------------------------------------------------

    def _bootstrap_cache_path(self) -> Path:
        """Return path to ~/.memra/bootstrap-cache.json (tests patch $HOME)."""
        base = Path.home() / ".memra"
        base.mkdir(parents=True, exist_ok=True)
        return base / "bootstrap-cache.json"

    def _bootstrap_ttl_seconds(self) -> int:
        raw = os.environ.get("MEMRA_BOOTSTRAP_TTL", "60")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 60

    def _read_bootstrap_cache(self) -> dict:
        path = self._bootstrap_cache_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8")) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_bootstrap_cache(self, cache: dict) -> None:
        path = self._bootstrap_cache_path()
        try:
            path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write bootstrap cache: %s", exc)

    def cloud_bootstrap(
        self,
        namespace: str,
        tenant_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Fetch cloud bootstrap memories for this namespace+agent.

        Returns [] on any failure (sync disabled, no agent_id, HTTP error,
        network error). Never raises — bootstrap() must degrade to local-only.

        Caches per (namespace, agent_id, tenant_id) to
        ``~/.memra/bootstrap-cache.json`` with TTL from ``MEMRA_BOOTSTRAP_TTL``
        (default 60s).
        """
        if not self.is_sync_enabled(namespace):
            return []

        config = self._conn.execute(
            "SELECT * FROM sync_cursors WHERE namespace = ?", (namespace,)
        ).fetchone()
        if config is None:
            return []

        config_keys = config.keys()
        agent_id = config["agent_id"] if "agent_id" in config_keys else None
        if not agent_id:
            logger.warning(
                "cloud bootstrap skipped: no agent_id stored for namespace %r. "
                "Re-run `memra sync enable` with --agent-id to enable cloud merge.",
                namespace,
            )
            return []

        api_url = config["cloud_api_url"]
        api_key = getattr(self, "_api_keys", {}).get(namespace, self._api_key or "")

        cache_key = f"{namespace}|{agent_id}|{tenant_id or ''}"
        ttl = self._bootstrap_ttl_seconds()
        cache = self._read_bootstrap_cache()
        entry = cache.get(cache_key)
        now = time.time()
        if entry and ttl > 0:
            fetched_at = entry.get("fetched_at", 0)
            if now - fetched_at < ttl:
                return list(entry.get("memories") or [])

        params: dict[str, str | int] = {"project_id": namespace}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if limit:
            params["max_memories"] = limit

        try:
            response = httpx.get(
                f"{api_url}/agents/{agent_id}/bootstrap",
                params=params,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # broad: httpx + json + anything else
            logger.warning(
                "cloud bootstrap failed for namespace=%r (%s); degrading to local-only.",
                namespace,
                exc,
            )
            return []

        data = payload.get("data") if isinstance(payload, dict) else None
        memories = []
        if isinstance(data, dict):
            memories = data.get("memories") or []
        elif isinstance(payload, dict):
            # Be lenient: accept unwrapped {memories: [...]} shape too.
            memories = payload.get("memories") or []

        if not isinstance(memories, list):
            memories = []

        cache[cache_key] = {
            "fetched_at": now,
            "memories": memories,
        }
        self._write_bootstrap_cache(cache)

        return memories
