"""Bootstrap endpoint — returns highest-importance memories for agent context."""

from __future__ import annotations

from fastapi import APIRouter, Request

from memra_local.models import BootstrapRequest

router = APIRouter(tags=["bootstrap"])


@router.post("/bootstrap")
def bootstrap(body: BootstrapRequest, request: Request):
    """Return top memories sorted by importance DESC for agent bootstrap."""
    service = request.app.state.service
    results = service.bootstrap(
        namespace=body.project_id,
        tenant_id=body.tenant_id,
        limit=body.limit,
    )
    return results
