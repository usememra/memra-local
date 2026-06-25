"""MCP stdio server for memra-local.

Exposes 18 Memra tools via FastMCP for IDE integration
(Claude Code, Cursor, Zed). Uses the same MemoryService as
the HTTP API via the shared service factory.

Tool surface (v0.6+): the primary verbs are `memra_remember` (write)
and `memra_recall` (read), matching the SaaS v4.3 canonical names.
All 16 original tool names (memra_add, memra_search, etc.) remain
callable — zero-break for existing users.

CRITICAL: No print() or logging to stdout -- MCP stdio transport
uses stdout for JSON-RPC messages. All logging must use stderr.
"""

from __future__ import annotations

import sys
import logging
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr only
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("memra.mcp")

mcp_app = FastMCP("memra-local")

# Module-level service singleton (initialized by init_service or set directly in tests)
_service = None
_scope = "auto"


def init_service(scope: str = "auto") -> None:
    """Initialize the global MemoryService via factory."""
    global _service, _scope
    from memra_local.services.factory import create_service

    _scope = scope
    _service = create_service(scope=scope)
    logger.info("Memra MCP service initialized (scope=%s)", scope)


def _get_service():
    """Get the initialized service, raising if not ready."""
    if _service is None:
        raise RuntimeError("MCP service not initialized. Call init_service() first.")
    return _service


# ------------------------------------------------------------------
# Tool: memra_add
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_add(
    content: str,
    namespace: str,
    type: str = "fact",
    importance: int = 5,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store a new memory. Content is deduplicated per namespace."""
    svc = _get_service()
    request = {
        "content": content,
        "tenant_id": "local",
        "project_id": namespace,
        "type": type,
        "importance": importance,
        "tags": tags or [],
        "metadata": metadata,
    }
    memory_data, is_duplicate = svc.add(request)
    if is_duplicate:
        memory_data["_duplicate"] = True
    return memory_data


# ------------------------------------------------------------------
# Tool: memra_get
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_get(memory_id: str) -> dict:
    """Retrieve a specific memory by its ID."""
    svc = _get_service()
    memory = svc.get(memory_id)
    if memory is None:
        return {"error": f"Memory not found: {memory_id}"}
    return memory


# ------------------------------------------------------------------
# Tool: memra_search
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_search(
    query: str,
    namespace: str,
    type: str | None = None,
    limit: int = 10,
) -> dict:
    """Search memories using semantic recall (embedding similarity + importance scoring)."""
    svc = _get_service()
    results, meta = svc.recall(
        query=query,
        namespace=namespace,
        tenant_id="local",
        type_=type,
        limit=limit,
    )
    return {"memories": results, "meta": meta}


# ------------------------------------------------------------------
# Tool: memra_list
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_list(
    namespace: str,
    type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List memories in a namespace with optional type filter and pagination."""
    svc = _get_service()
    items, total = svc.list_(
        namespace=namespace,
        tenant_id="local",
        type_=type,
        limit=limit,
        offset=offset,
    )
    return {"memories": items, "total": total}


# ------------------------------------------------------------------
# Tool: memra_bootstrap
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_bootstrap(namespace: str, limit: int = 20) -> dict:
    """Get priority-ordered memories for agent bootstrap context."""
    svc = _get_service()
    results = svc.bootstrap(
        namespace=namespace,
        tenant_id="local",
        limit=limit,
    )
    return {"memories": results}


# ------------------------------------------------------------------
# Tool: memra_get_health
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_get_health(namespace: str | None = None) -> dict:
    """Get health stats: memory count, scope, disk usage, embedding coverage."""
    svc = _get_service()

    # Count memories
    if namespace:
        _, total = svc.list_(namespace=namespace, tenant_id="local", limit=0, offset=0)
    else:
        _, total = svc.list_(limit=0, offset=0)

    # Disk usage (storage dir size in MB)
    storage_dir = svc.store._base_dir
    disk_bytes = sum(f.stat().st_size for f in storage_dir.rglob("*") if f.is_file())
    disk_mb = round(disk_bytes / (1024 * 1024), 2)

    return {
        "memory_count": total,
        "scope": _scope,
        "disk_usage_mb": disk_mb,
    }


# ------------------------------------------------------------------
# Tool: memra_add_decision
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_add_decision(
    content: str,
    namespace: str,
    context: str | None = None,
) -> dict:
    """Store an irreversible decision with high importance (8/10)."""
    svc = _get_service()
    full_content = content
    if context:
        full_content = f"Context: {context}\n\nDecision: {content}"

    request = {
        "content": full_content,
        "tenant_id": "local",
        "project_id": namespace,
        "type": "decision",
        "importance": 8,
        "tags": ["decision"],
    }
    memory_data, _ = svc.add(request)
    return memory_data


# ------------------------------------------------------------------
# Tool: memra_add_pattern
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_add_pattern(
    name: str,
    namespace: str,
    steps: list[str] | None = None,
    gotchas: list[str] | None = None,
) -> dict:
    """Store a reusable workflow pattern with formatted steps and gotchas."""
    svc = _get_service()

    # Format content from name, steps, gotchas
    parts = [f"Pattern: {name}"]
    if steps:
        parts.append("\nSteps:")
        for i, step in enumerate(steps, 1):
            parts.append(f"  {i}. {step}")
    if gotchas:
        parts.append("\nGotchas:")
        for gotcha in gotchas:
            parts.append(f"  - {gotcha}")

    content = "\n".join(parts)

    request = {
        "content": content,
        "tenant_id": "local",
        "project_id": namespace,
        "type": "pattern",
        "importance": 7,
        "tags": ["pattern"],
    }
    memory_data, _ = svc.add(request)
    return memory_data


# ------------------------------------------------------------------
# Tool: memra_refresh_memory
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_refresh_memory(memory_id: str) -> dict:
    """Touch a memory to update its updated_at timestamp (prevents staleness)."""
    svc = _get_service()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = svc.update(memory_id, {"updated_at": now})
    if updated is None:
        return {"error": f"Memory not found: {memory_id}"}
    return updated


# ------------------------------------------------------------------
# Tool: memra_promote
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_promote(memory_id: str, promoted_by: str | None = None) -> dict:
    """Promote a memory from 'proposed' to 'verified' trust state.

    Use when an auto-captured memory has been reviewed and confirmed accurate.
    Verified memories are trusted and surface in normal recall; proposed ones are
    held back until reviewed. Mirrors POST /memories/{id}/promote on the cloud API.

    Args:
        memory_id: The memory to promote.
        promoted_by: Optional actor recorded in metadata for audit (e.g. "user").

    Returns:
        The updated memory dict on success. Returns {"error": ...} if the memory
        does not exist or is not currently in 'proposed' state.
    """
    import json as _json

    svc = _get_service()
    memory = svc.get(memory_id)
    if memory is None:
        return {"error": f"Memory not found: {memory_id}"}

    metadata = memory.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = _json.loads(metadata)

    current_state = metadata.get("trust_state", "verified")
    if current_state != "proposed":
        return {
            "error": (
                f"Memory is in trust_state '{current_state}'; "
                "only 'proposed' memories can be promoted."
            )
        }

    metadata["trust_state"] = "verified"
    metadata["verified_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    if promoted_by:
        metadata["verified_by"] = promoted_by

    updated = svc.update(memory_id, {"metadata": metadata})
    if updated is None:
        return {"error": f"Failed to update memory: {memory_id}"}
    return updated


# ------------------------------------------------------------------
# Tool: memra_supersede
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_supersede(old_memory_id: str, content: str, namespace: str | None = None) -> dict:
    """Supersede an existing memory with new content. Returns the new memory and old ID."""
    from memra_local.exceptions import ConcurrentModificationError

    svc = _get_service()
    try:
        new_mem, _old_mem = svc.supersede(old_memory_id, content)
        return {"new_memory": new_mem, "superseded_memory_id": old_memory_id}
    except ValueError as e:
        return {"error": str(e)}
    except ConcurrentModificationError:
        return {"error": "Memory was modified concurrently. Re-read and retry."}


# ------------------------------------------------------------------
# Tool: memra_history
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_history(memory_id: str) -> dict:
    """View the supersession chain for a memory (oldest to newest)."""
    svc = _get_service()
    chain = svc.get_chain(memory_id)
    if not chain:
        return {"error": f"Memory not found: {memory_id}"}
    return {"chain": chain, "length": len(chain)}


# ------------------------------------------------------------------
# Tool: memra_sync_enable
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_sync_enable(
    namespace: str,
    api_key: str,
    api_url: str = "https://usememra.com/api/v1",
    mode: str = "local_private",
) -> dict:
    """Enable cloud sync for a namespace.

    Args:
        mode: PII sharing mode. One of 'local_private' (default — pull-only,
            push is disabled), 'shared_masked' (content is PII-masked before
            push), or 'shared_raw' (raw content; requires Team/Admin tier).
    """
    from memra_local.services.sync_service import SyncService

    svc = _get_service()
    if mode not in SyncService.VALID_PII_MODES:
        return {
            "error": (
                f"Invalid mode '{mode}'. "
                f"Valid modes: {', '.join(SyncService.VALID_PII_MODES)}"
            )
        }
    svc.sync_service.enable(namespace, api_key, api_url, pii_mode=mode)
    result = {"status": "enabled", "namespace": namespace, "mode": mode}
    if mode == "local_private":
        result["note"] = (
            "Push is disabled in local_private mode (pull-only). "
            "Re-run memra_sync_enable with mode='shared_masked' or "
            "mode='shared_raw' to allow pushing."
        )
    return result


# ------------------------------------------------------------------
# Tool: memra_sync_push
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_sync_push(namespace: str) -> dict:
    """Push local changes to cloud for a synced namespace."""
    svc = _get_service()
    result = svc.sync_service.push(namespace)
    return result.model_dump()


# ------------------------------------------------------------------
# Tool: memra_sync_pull
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_sync_pull(namespace: str) -> dict:
    """Pull remote changes from cloud for a synced namespace."""
    svc = _get_service()
    result = svc.sync_service.pull(namespace, svc)
    return result.model_dump()


# ------------------------------------------------------------------
# Tool: memra_migrate
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_migrate(
    api_key: str,
    api_url: str = "https://usememra.com/api/v1",
    dry_run: bool = False,
) -> str:
    """Migrate all local memories to cloud account.

    Uploads local memories in batches of 50 via the cloud batch endpoint.
    PII masking is applied automatically for safety.

    Args:
        api_key: Cloud API key (memra_live_...)
        api_url: Cloud API base URL
        dry_run: If True, count memories without uploading
    """
    from memra_local.services.pii_client import PiiClient
    from memra_local.services.migration_service import MigrationService

    svc = _get_service()
    pii_client = PiiClient(api_url=api_url, api_key=api_key)
    migration = MigrationService(api_url=api_url, api_key=api_key, pii_client=pii_client)

    result = migration.migrate(index=svc.index, store=svc.store, dry_run=dry_run)

    if result.dry_run:
        return f"Dry run: {result.total} memories would be migrated"
    parts = [f"Migration complete: {result.migrated} migrated, {result.skipped} duplicates skipped"]
    if result.failed > 0:
        parts.append(f", {result.failed} failed")
    if result.errors:
        parts.append(f"\nErrors: {'; '.join(result.errors)}")
    return "".join(parts)


# ------------------------------------------------------------------
# Tool: memra_remember (v4.3 canonical write verb)
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_remember(
    content: str | None = None,
    namespace: str | None = None,
    type: str = "fact",
    importance: int = 5,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    entries: list[dict] | None = None,
    context: str | None = None,
    steps: list[str] | None = None,
    gotchas: list[str] | None = None,
    title: str | None = None,
) -> dict:
    """Primary memory-write verb. Mirrors SaaS v4.3 memra_remember.

    Dispatches by argument shape:
    - entries set → bulk add (returns {"memories": [...], "count": N})
    - type='decision' → memra_add_decision(content, namespace, context)
    - type='pattern'  → memra_add_pattern(title, namespace, steps, gotchas)
    - otherwise       → memra_add(content, namespace, type, importance, tags, metadata)
    """
    if entries:
        results = [
            memra_add(
                content=e.get("content"),
                namespace=e.get("namespace", namespace),
                type=e.get("type", "fact"),
                importance=e.get("importance", 5),
                tags=e.get("tags"),
                metadata=e.get("metadata"),
            )
            for e in entries
        ]
        return {"memories": results, "count": len(results)}

    if type == "decision":
        return memra_add_decision(
            content=content,
            namespace=namespace,
            context=context,
        )

    if type == "pattern":
        return memra_add_pattern(
            name=title or content or "Untitled pattern",
            namespace=namespace,
            steps=steps,
            gotchas=gotchas,
        )

    return memra_add(
        content=content,
        namespace=namespace,
        type=type,
        importance=importance,
        tags=tags,
        metadata=metadata,
    )


# ------------------------------------------------------------------
# Tool: memra_recall (v4.3 canonical read verb — alias of memra_search)
# ------------------------------------------------------------------
@mcp_app.tool()
def memra_recall(
    query: str,
    namespace: str,
    type: str | None = None,
    limit: int = 10,
) -> dict:
    """Primary memory-read verb. Alias of memra_search, mirrors SaaS v4.3."""
    return memra_search(query=query, namespace=namespace, type=type, limit=limit)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def run_mcp(scope: str = "auto") -> None:
    """Initialize service and start MCP stdio transport."""
    init_service(scope=scope)
    logger.info("Starting Memra MCP stdio server...")
    mcp_app.run(transport="stdio")
