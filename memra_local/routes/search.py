"""Search and recall endpoints — FTS5 text search (semantic upgrade in Phase 47)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from memra_local.models import RecallRequest, SearchRequest

router = APIRouter(tags=["search"])


@router.post("/memories/search")
def search_memories(body: SearchRequest, request: Request):
    """Full-text search for memories using FTS5."""
    service = request.app.state.service
    results = service.search(
        query=body.query,
        namespace=body.project_id,
        tenant_id=body.tenant_id,
        type_=body.type,
        importance_min=body.importance_min,
        limit=body.limit,
        offset=body.offset,
    )
    return results


@router.post("/memories/recall")
def recall_memories(body: RecallRequest, request: Request):
    """Recall memories using text match (semantic upgrade in Phase 47)."""
    service = request.app.state.service
    results, meta = service.recall(
        query=body.query,
        namespace=body.project_id,
        tenant_id=body.tenant_id,
        type_=body.type,
        importance_min=body.importance_min,
        limit=body.limit,
    )
    return {"data": results, "meta": meta}
