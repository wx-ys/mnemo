"""Mnemo REST API server — FastAPI application.

Provides a RESTful HTTP API for all Mnemo operations.  OpenAPI docs
are auto-generated at ``/docs`` (Swagger) and ``/redoc`` (ReDoc).

Usage::

    mnemo serve                    # start on default host:port
    mnemo serve --port 8080        # custom port
    mnemo serve --reload           # development mode
    mnemo serve --host 0.0.0.0     # listen on all interfaces

Or programmatically::

    import uvicorn
    uvicorn.run("mnemo.api.server:app", host="127.0.0.1", port=8765)
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from mnemo.api import MnemoAPI, SearchMode
from mnemo.api.ask_types import AskResponse
from mnemo.api.types import (
    FileContext,
    FileInfo,
    KnowledgeBaseStats,
    SearchResult,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mnemo API",
    version="1.0.0",
    description="REST API for Mnemo — a file-based personal knowledge base",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Root — redirect to docs
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    """Redirect to OpenAPI docs."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

def get_api() -> MnemoAPI:
    """Dependency: get or create a MnemoAPI instance (cached per request)."""
    data_dir = os.environ.get("MNEMO_DATA_DIR", "~/mnemo-data")
    return MnemoAPI(data_dir)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    mode: str = Field("hybrid", description="Search mode: hybrid, vector, keyword")
    keys: list[str] | None = Field(None, description="Limit to key scopes")
    file_types: list[str] | None = Field(None, description="Limit to file extensions")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    grounded: bool = Field(True, description="Strictly ground in KB content")
    limit: int = Field(10, ge=1, le=20, description="Maximum source chunks")


class UpdateRequest(BaseModel):
    keys: list[str] | None = Field(None, description="Replace all keys")
    tags: list[str] | None = Field(None, description="Replace all tags")
    note: str | None = Field(None, description="Update user note")
    category: str | None = Field(None, description="Update category")


class ExportRequest(BaseModel):
    file_type: str | None = Field(None, description="Export only this type")
    keys: list[str] | None = Field(None, description="Export files matching these keys")
    after: str | None = Field(None, description="ISO 8601 date filter")


# ---------------------------------------------------------------------------
# Routes — Files
# ---------------------------------------------------------------------------

@app.post("/api/v1/files", tags=["Files"])
def add_file(
    source: str = Query(..., description="Path to source file"),
    move: bool = Query(False, description="Move instead of copy"),
    keys: list[str] | None = Query(None, description="Hierarchical keys"),
    tags: list[str] | None = Query(None, description="Flat tags"),
    note: str = Query("", description="Initial user note"),
    overwrite: bool = Query(False, description="Overwrite duplicate"),
    api: MnemoAPI = Depends(get_api),
) -> FileInfo:
    """Add a file to the knowledge base."""
    return api.add(
        source=source, move=move, keys=keys, tags=tags,
        note=note, overwrite=overwrite,
    )


@app.get("/api/v1/files", tags=["Files"])
def list_files(
    file_type: str | None = Query(None, description="Filter by extension"),
    keys: list[str] | None = Query(None, description="Filter by keys"),
    tags: list[str] | None = Query(None, description="Filter by tags"),
    date_from: str | None = Query(None, description="ISO 8601 lower bound"),
    date_to: str | None = Query(None, description="ISO 8601 upper bound"),
    sort_by: str = Query("added_at", description="Sort column"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    api: MnemoAPI = Depends(get_api),
) -> list[FileInfo]:
    """List files with filters and pagination."""
    return api.list_files(
        file_type=file_type, keys=keys, tags=tags,
        date_from=date_from, date_to=date_to,
        sort_by=sort_by, limit=limit, offset=offset,
    )


@app.get("/api/v1/files/{file_id}", tags=["Files"])
def get_file(
    file_id: str,
    api: MnemoAPI = Depends(get_api),
) -> FileInfo:
    """Get file information by ID or filename."""
    resolved = api.resolve_file_ref(file_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return api.get(resolved)


@app.get("/api/v1/files/{file_id}/context", tags=["Files"])
def get_file_context(
    file_id: str,
    api: MnemoAPI = Depends(get_api),
) -> FileContext:
    """Get full file context (md, wiki, entities, notes)."""
    resolved = api.resolve_file_ref(file_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return api.get_context(resolved)


@app.patch("/api/v1/files/{file_id}", tags=["Files"])
def update_file(
    file_id: str,
    body: UpdateRequest,
    api: MnemoAPI = Depends(get_api),
) -> FileInfo:
    """Update file metadata."""
    resolved = api.resolve_file_ref(file_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return api.update(resolved, keys=body.keys, tags=body.tags,
                      note=body.note, category=body.category)


@app.delete("/api/v1/files/{file_id}", tags=["Files"])
def remove_file(
    file_id: str,
    api: MnemoAPI = Depends(get_api),
) -> dict:
    """Soft-delete a file (moves to trash)."""
    resolved = api.resolve_file_ref(file_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
    return api.remove(resolved)


# ---------------------------------------------------------------------------
# Routes — Search & Ask
# ---------------------------------------------------------------------------

@app.post("/api/v1/search", tags=["Search"])
def search(
    body: SearchRequest,
    api: MnemoAPI = Depends(get_api),
) -> list[SearchResult]:
    """Search the knowledge base."""
    mode_map = {
        "hybrid": SearchMode.HYBRID,
        "vector": SearchMode.VECTOR,
        "keyword": SearchMode.KEYWORD,
    }
    mode = mode_map.get(body.mode, SearchMode.HYBRID)
    return api.search(
        query=body.query, mode=mode, keys=body.keys,
        file_types=body.file_types, limit=body.limit,
    )


@app.post("/api/v1/ask", tags=["Search"])
def ask(
    body: AskRequest,
    api: MnemoAPI = Depends(get_api),
) -> AskResponse:
    """Ask a question — RAG answer with citations."""
    return api.ask(
        question=body.question, grounded=body.grounded,
        limit=body.limit,
    )


# ---------------------------------------------------------------------------
# Routes — Management
# ---------------------------------------------------------------------------

@app.get("/api/v1/stats", tags=["Management"])
def stats(
    api: MnemoAPI = Depends(get_api),
) -> KnowledgeBaseStats:
    """Get knowledge base statistics."""
    return api.stats()


@app.post("/api/v1/export", tags=["Management"])
def export_kb(
    dest: str = Query(..., description="Destination path (.tar.gz)"),
    body: ExportRequest | None = None,
    api: MnemoAPI = Depends(get_api),
) -> dict:
    """Export the knowledge base."""
    if body is None:
        body = ExportRequest()
    archive = api.export_kb(
        dest=dest, file_type=body.file_type,
        keys=body.keys, after=body.after,
    )
    return {"path": str(archive)}


@app.post("/api/v1/import", tags=["Management"])
def import_kb(
    source: str = Query(..., description="Source path (.tar.gz)"),
    dry_run: bool = Query(False, description="Preview only"),
    api: MnemoAPI = Depends(get_api),
) -> dict:
    """Import a knowledge base archive."""
    report = api.import_kb(source=source, dry_run=dry_run)
    return {
        "imported": report.imported,
        "skipped": report.skipped,
        "errors": report.errors,
    }
