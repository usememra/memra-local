"""Pydantic models for memra-local — wire-compatible with Python SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Memory(BaseModel):
    """Full memory object returned by add, get, update endpoints."""

    id: str
    content: str | None = None
    tenant_id: str
    project_id: str
    type: str
    importance: int = 5
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    metadata: dict[str, Any] | None = None
    embedding_status: str = "none"
    expires_at: str | None = None
    confidence: float = 1.0
    staleness_score: int = 0
    last_accessed_at: str | None = None
    status: str = "active"
    superseded_by: str | None = None
    ttl_days: int | None = None
    soft_ttl_days: int | None = None
    created_at: str
    updated_at: str


class MemoryListItem(BaseModel):
    """Memory metadata returned by list endpoint (no content field)."""

    id: str
    tenant_id: str
    project_id: str
    type: str
    importance: int = 5
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    metadata: dict[str, Any] | None = None
    embedding_status: str = "none"
    expires_at: str | None = None
    confidence: float = 1.0
    staleness_score: int = 0
    last_accessed_at: str | None = None
    status: str = "active"
    superseded_by: str | None = None
    ttl_days: int | None = None
    soft_ttl_days: int | None = None
    created_at: str
    updated_at: str


class MemoryList(BaseModel):
    """Paginated list of memories."""

    memories: list[MemoryListItem]
    total: int
    limit: int
    offset: int
    has_more: bool


class RecallMemory(BaseModel):
    """Single memory item in recall results."""

    id: str
    content: str
    score: float
    type: str
    importance: int
    tags: list[str] = Field(default_factory=list)
    created_at: str


class RecallMeta(BaseModel):
    """Metadata about the recall query execution."""

    total_candidates: int
    returned: int
    scoring: str
    query_cached: bool
    response_cached: bool


class RecallResult(BaseModel):
    """Recall search results with data and metadata."""

    data: list[RecallMemory]
    meta: RecallMeta


class AddMemoryRequest(BaseModel):
    """Request to add a new memory."""

    content: str
    tenant_id: str = "local"
    project_id: str = "default"
    type: str = "fact"
    importance: int = 5
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("tenant_id", "project_id")
    @classmethod
    def validate_scope_segment(cls, value: str) -> str:
        if not value or any(part in value for part in ("..", "/", "\\", "\x00")):
            raise ValueError("must not contain path traversal characters")
        return value


class SearchRequest(BaseModel):
    """Request for FTS5 text search."""

    query: str
    tenant_id: str = "local"
    project_id: str = "default"
    type: str | None = None
    importance_min: int | None = None
    limit: int = 20
    offset: int = 0

    @field_validator("tenant_id", "project_id")
    @classmethod
    def validate_scope_segment(cls, value: str) -> str:
        if not value or any(part in value for part in ("..", "/", "\\", "\x00")):
            raise ValueError("must not contain path traversal characters")
        return value


class RecallRequest(BaseModel):
    """Request for semantic recall."""

    query: str
    tenant_id: str = "local"
    project_id: str = "default"
    limit: int = 10
    type: str | None = None
    importance_min: int | None = None
    scoring: str | None = None

    @field_validator("tenant_id", "project_id")
    @classmethod
    def validate_scope_segment(cls, value: str) -> str:
        if not value or any(part in value for part in ("..", "/", "\\", "\x00")):
            raise ValueError("must not contain path traversal characters")
        return value


class BootstrapRequest(BaseModel):
    """Request for agent bootstrap context."""

    tenant_id: str = "local"
    project_id: str = "default"
    limit: int = 20

    @field_validator("tenant_id", "project_id")
    @classmethod
    def validate_scope_segment(cls, value: str) -> str:
        if not value or any(part in value for part in ("..", "/", "\\", "\x00")):
            raise ValueError("must not contain path traversal characters")
        return value


# -------------------------------------------------------------------
# Sync models
# -------------------------------------------------------------------


class SyncEvent(BaseModel):
    """A single sync event recorded locally."""

    id: str
    namespace: str
    event_type: str  # memory_created, memory_updated, memory_deleted
    memory_id: str
    payload: dict[str, Any]
    base_revision: int | None = None
    pushed_at: str | None = None
    created_at: str


class SyncConfig(BaseModel):
    """Configuration for a sync-enabled namespace."""

    namespace: str
    cloud_api_url: str = "https://usememra.com/api/v1"
    remote_cursor: int = 0
    last_synced_at: str | None = None
    enabled_at: str
    pii_mode: str = "local_private"


class SyncPushResult(BaseModel):
    """Result of a push operation."""

    pushed: int
    conflicts: list[dict[str, str]] = Field(default_factory=list)
    cursor: int | None = None
    error: str | None = None


class SyncPullResult(BaseModel):
    """Result of a pull operation."""

    applied: int
    cursor: int
    has_more: bool
    error: str | None = None


class SupersedeRequest(BaseModel):
    """Request to supersede an existing memory with new content."""

    content: str
    type: str | None = None
    importance: int | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class MigrateResult(BaseModel):
    """Result of a local->cloud migration."""

    total: int
    migrated: int
    skipped: int  # duplicates
    failed: int
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False
