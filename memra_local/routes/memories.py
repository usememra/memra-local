"""CRUD endpoints for memories — wire-compatible with cloud Memra API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from memra_local.exceptions import ConcurrentModificationError
from memra_local.models import AddMemoryRequest, SupersedeRequest

router = APIRouter(tags=["memories"])


@router.post("/memories", status_code=201)
def add_memory(body: AddMemoryRequest, request: Request):
    """Create a new memory. Returns 200 for duplicates, 201 for new."""
    service = request.app.state.service
    data, is_duplicate = service.add(body.model_dump())

    if is_duplicate:
        return JSONResponse(content=data, status_code=200)

    return JSONResponse(content=data, status_code=201)


@router.get("/memories")
def list_memories(
    request: Request,
    tenant_id: str | None = None,
    project_id: str | None = None,
    type: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List memories with optional filters."""
    service = request.app.state.service
    memories, total = service.list_(
        namespace=project_id,
        tenant_id=tenant_id,
        type_=type,
        limit=limit,
        offset=offset,
    )

    return {
        "memories": memories,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
    }


@router.get("/memories/{memory_id}")
def get_memory(memory_id: str, request: Request):
    """Get a single memory by ID."""
    service = request.app.state.service
    data = service.get(memory_id)
    if data is None:
        return JSONResponse(
            content={"code": "not_found", "message": f"Memory {memory_id} not found"},
            status_code=404,
        )
    return data


@router.patch("/memories/{memory_id}")
def update_memory(
    memory_id: str,
    body: dict[str, Any],
    request: Request,
):
    """Update an existing memory."""
    service = request.app.state.service
    data = service.update(memory_id, body)
    if data is None:
        return JSONResponse(
            content={"code": "not_found", "message": f"Memory {memory_id} not found"},
            status_code=404,
        )
    return data


@router.post("/memories/{memory_id}/supersede", status_code=201)
def supersede_memory(memory_id: str, body: SupersedeRequest, request: Request):
    """Supersede a memory with new content. Returns 201 with new memory."""
    service = request.app.state.service

    # Build kwargs from non-None optional fields
    kwargs = {}
    if body.type is not None:
        kwargs["type"] = body.type
    if body.importance is not None:
        kwargs["importance"] = body.importance
    if body.tags is not None:
        kwargs["tags"] = body.tags
    if body.metadata is not None:
        kwargs["metadata"] = body.metadata

    try:
        new_mem, _old_mem = service.supersede(memory_id, body.content, **kwargs)
        return JSONResponse(content={"data": new_mem}, status_code=201)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            return JSONResponse(
                content={"code": "not_found", "message": msg},
                status_code=404,
            )
        if "already superseded" in msg.lower():
            return JSONResponse(
                content={"code": "conflict", "message": msg},
                status_code=409,
            )
        return JSONResponse(
            content={"code": "bad_request", "message": msg},
            status_code=400,
        )
    except ConcurrentModificationError as e:
        return JSONResponse(
            content={"code": "conflict", "message": str(e)},
            status_code=409,
        )


@router.get("/memories/{memory_id}/chain")
def get_memory_chain(memory_id: str, request: Request):
    """Get the supersession chain for a memory."""
    service = request.app.state.service
    chain = service.get_chain(memory_id)
    if not chain:
        return JSONResponse(
            content={"code": "not_found", "message": f"Memory {memory_id} not found"},
            status_code=404,
        )
    return {"data": chain, "length": len(chain)}


@router.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str, request: Request):
    """Delete a memory. Returns 204 on success, 404 if not found."""
    service = request.app.state.service
    deleted = service.delete(memory_id)
    if not deleted:
        return JSONResponse(
            content={"code": "not_found", "message": f"Memory {memory_id} not found"},
            status_code=404,
        )
    return Response(status_code=204)
