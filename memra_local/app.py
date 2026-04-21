"""FastAPI application factory with lifespan for memra-local."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from memra_local import __version__
from memra_local.config import resolve_scope
from memra_local.routes import bootstrap, memories, search
from memra_local.services.factory import create_service


def create_app(
    scope: str = "auto",
    storage_dir: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        scope: Storage scope -- "global", "project", or "auto".
        storage_dir: Override storage directory (used by tests).

    Returns:
        Configured FastAPI app with service on app.state.service.
    """
    resolved_scope = resolve_scope(scope) if storage_dir is None else scope

    # Use shared factory for service creation (same as MCP server)
    service = create_service(scope=scope, storage_dir=storage_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Index is already initialized by factory; just handle shutdown."""
        yield
        service.index.close()

    app = FastAPI(
        title="Memra Local",
        version=__version__,
        description="Local memory server for AI agents",
        lifespan=lifespan,
    )

    # Store service on app state for route access
    app.state.service = service

    # Include routers with /v1 prefix
    app.include_router(memories.router, prefix="/v1")
    app.include_router(search.router, prefix="/v1")
    app.include_router(bootstrap.router, prefix="/v1")

    # Health endpoint (no prefix)
    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "version": __version__,
            "scope": resolved_scope,
        }

    return app
