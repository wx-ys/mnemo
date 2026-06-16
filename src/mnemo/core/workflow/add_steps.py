"""Add workflow step implementations — file ingestion business logic.

Migrated from the old ``core/pipeline.py``.  Each step is a registered
FunctionStep that the WorkflowEngine can execute in DAG order.

Steps:
    validate → copy → metadata → markdown → wiki → entity_extract → embed → index
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from mnemo.api.types import FileInfo, FileStatus
from mnemo.core.workflow.context import WorkflowContext
from mnemo.core.workflow.events import WorkflowEvent
from mnemo.core.workflow.step import StepRegistry


if TYPE_CHECKING:
    from mnemo.core.kb import KnowledgeBase


def _get_kb(ctx: WorkflowContext) -> KnowledgeBase:
    return ctx.kb


# ============================================================================
# Step 1: Validate
# ============================================================================


@StepRegistry.register_function("validate_file")
async def validate_file(ctx: WorkflowContext) -> dict[str, Any]:
    """Validate source file and resolve file classification."""
    kb = _get_kb(ctx)
    source_path = ctx.get_input("source_path")

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    file_ext = source_path.suffix.lower().lstrip(".")
    cat_name, file_type = kb._resolve_file_classification(file_ext)

    from mnemo.core.interfaces import IParser, ITemplate
    parser = IParser.resolve(file_ext, cat_name)
    template = ITemplate.resolve(file_ext, cat_name)

    file_hash = kb._compute_hash(source_path)
    overwrite = ctx.config.get("overwrite", False)

    if not overwrite:
        existing_id = kb.indexer.file_exists_by_hash(file_hash)
        if existing_id:
            return {"_duplicate_id": existing_id}

    chunk_dir = kb._get_current_chunk(file_type)
    raw_dest = (
        kb.data_dir / "raw" / cat_name / file_type / chunk_dir
        / source_path.name
    )
    if raw_dest.exists() and not overwrite:
        raise FileExistsError(
            f"A file named '{source_path.name}' already exists in the "
            f"knowledge base with different content. Use --overwrite."
        )

    auto_md = kb._resolve_flag("auto_md", ctx.config.get("auto_md"), file_type)
    auto_wiki = kb._resolve_flag("auto_wiki", ctx.config.get("auto_wiki"), file_type)
    auto_embed = kb._resolve_flag("auto_embed", ctx.config.get("auto_embed"), file_type)

    # -- Diagnostic ----------------------------------------------------------
    if ctx.diagnostic and ctx.diagnostic.enabled:
        ctx.emit("step.progress", step_name="validate_file",
                 data={"_diagnostic": {
                     "source_path": str(source_path),
                     "file_size": source_path.stat().st_size,
                     "file_ext": file_ext,
                     "category": cat_name,
                     "file_type": file_type,
                     "file_hash": file_hash,
                     "parser": parser.name if parser else "",
                     "template": template.name if template else "",
                     "auto_md": auto_md,
                     "auto_wiki": auto_wiki,
                     "auto_embed": auto_embed,
                 }})

    return {
        "cat_name": cat_name,
        "file_type": file_type,
        "file_hash": file_hash,
        "parser": parser,
        "template": template,
        "chunk_dir": chunk_dir,
        "raw_dest": raw_dest,
        "auto_md": auto_md,
        "auto_wiki": auto_wiki,
        "auto_embed": auto_embed,
    }


# ============================================================================
# Step 2: Copy
# ============================================================================


@StepRegistry.register_function("copy_file")
async def copy_file(ctx: WorkflowContext) -> dict[str, Any]:
    """Copy source file into raw/ directory."""
    raw_dest = ctx.data["raw_dest"]
    raw_dest.parent.mkdir(parents=True, exist_ok=True)

    if ctx.config.get("move", False):
        ctx.get_input("source_path").rename(raw_dest)
    else:
        shutil.copy2(ctx.get_input("source_path"), raw_dest)

    file_id = str(uuid.uuid4())

    # -- Diagnostic ----------------------------------------------------------
    if ctx.diagnostic and ctx.diagnostic.enabled:
        ctx.emit("step.progress", step_name="copy_file",
                 data={"_diagnostic": {
                     "file_id": file_id,
                     "dest": str(raw_dest),
                     "moved": ctx.config.get("move", False),
                 }})

    return {"file_id": file_id}


# ============================================================================
# Step 3: Metadata
# ============================================================================


@StepRegistry.register_function("create_metadata")
async def create_metadata(ctx: WorkflowContext) -> dict[str, Any]:
    """Create preliminary FileInfo and compute keys."""
    kb = _get_kb(ctx)
    source_path = ctx.get_input("source_path")
    tags = ctx.data.get("tags") or []
    keys = ctx.data.get("keys") or []
    note = ctx.config.get("note", "")

    meta = FileInfo(
        id=ctx.data["file_id"],
        file_type=ctx.data["file_type"],
        filename=source_path.name,
        file_size=source_path.stat().st_size,
        file_hash=ctx.data["file_hash"],
        raw_path=str(ctx.data["raw_dest"].relative_to(kb.data_dir)),
        source_path=str(source_path),
        tags=tags,
        keys=keys,
        category=ctx.data.get("cat_name", ""),
        custom={"note": note} if note else {},
    )

    all_keys = list(keys)
    auto_key = f"file_type::{ctx.data['file_type']}"
    if auto_key not in all_keys:
        all_keys = [auto_key] + all_keys

    # -- Diagnostic ----------------------------------------------------------
    if ctx.diagnostic and ctx.diagnostic.enabled:
        ctx.emit("step.progress", step_name="create_metadata",
                 data={"_diagnostic": {
                     "file_id": ctx.data["file_id"],
                     "file_type": ctx.data["file_type"],
                     "category": ctx.data.get("cat_name", ""),
                     "keys": all_keys,
                     "tags": tags,
                     "note": note,
                 }})

    return {
        "file_metadata": meta,
        "all_keys": all_keys,
    }


# ============================================================================
# Step 4: Parse to Markdown
# ============================================================================


@StepRegistry.register_function("parse_file_to_markdown")
async def parse_file_to_markdown(ctx: WorkflowContext) -> dict[str, Any]:
    """Parse raw file to Markdown and write to raw_md/."""
    kb = _get_kb(ctx)
    if not ctx.data.get("auto_md"):
        return {"md_content": "", "md_info": None}

    source_path = ctx.get_input("source_path")
    parser = ctx.data["parser"]
    md_content = parser.parse(ctx.data["raw_dest"])
    md_lines = md_content.splitlines()

    md_dest = (
        kb.data_dir / "raw_md" / ctx.data["cat_name"] / ctx.data["file_type"]
        / ctx.data["chunk_dir"] / f"{source_path.stem}.md"
    )
    md_dest.parent.mkdir(parents=True, exist_ok=True)
    md_dest.write_text(md_content, encoding="utf-8")

    # -- Diagnostic ----------------------------------------------------------
    if ctx.diagnostic and ctx.diagnostic.enabled:
        max_preview = ctx.diagnostic.max_text_preview_chars
        ctx.emit("step.progress", step_name="parse_file_to_markdown",
                 data={"_diagnostic": {
                     "parser": parser.name,
                     "md_preview": md_content[:max_preview],
                     "md_total_chars": len(md_content),
                     "md_lines": len(md_lines),
                     "md_size_bytes": len(md_content.encode("utf-8")),
                     "md_dest": str(md_dest.relative_to(kb.data_dir)),
                 }})

    return {
        "md_content": md_content,
        "md_info": {
            "status": "done",
            "generated_at": datetime.now(UTC).isoformat(),
            "parser": parser.name,
            "chars": len(md_content),
            "lines": len(md_lines),
            "file_size": len(md_content.encode("utf-8")),
        },
    }


# ============================================================================
# Step 5: Generate Wiki (LLM)
# ============================================================================


@StepRegistry.register_function("generate_wiki")
async def generate_wiki(ctx: WorkflowContext) -> dict[str, Any]:
    """Generate LLM wiki summary and write to raw_wiki/.

    Uses streaming (``agent.run_stream()``) for real-time CLI progress
    and captures actual token usage from pydantic-ai's ``RunUsage``.
    """
    kb = _get_kb(ctx)
    if not ctx.data.get("auto_wiki") or not ctx.data.get("md_content"):
        return {"wiki_content": "", "wiki_info": None}

    from mnemo.core.param_config import resolve_agent_config

    llm_config = resolve_agent_config()
    source_path = ctx.get_input("source_path")
    template = ctx.data.get("template")
    api_key = llm_config.get("api_key", "")

    if not api_key:
        return {
            "wiki_content": "",
            "wiki_info": {
                "status": "pending", "generated_at": "",
                "template": getattr(template, 'name', ''), "model": "",
                "chars": 0, "tokens_used": 0,
            },
        }

    try:
        # Build callbacks that emit events through the EventBus.
        # - Text chunks → stream.chunk (LogSink captures for JSON logs;
        #   CLISink writes at DEBUG level only, no terminal output).
        # - Thinking chunks → step.progress → LegacyProgressSink →
        #   on_progress → ProgressDisplay (Rich spinner inline).
        emitter = ctx.emitter
        run_id = ctx.run_id
        workflow_name = ctx.workflow_name
        step_name = "generate_wiki"
        _first_chunk_sent = False

        def on_chunk(chunk: str) -> None:
            nonlocal _first_chunk_sent
            try:
                emitter.emit_sync(
                    WorkflowEvent.stream_chunk(
                        run_id, workflow_name, step_name, chunk,
                        is_first=not _first_chunk_sent,
                    ),
                )
                _first_chunk_sent = True
            except Exception:
                pass  # fire-and-forget — never let stream events crash generation

        def on_thinking(thinking_text: str) -> None:
            """Emit LLM thinking/reasoning content as step.progress.

            Uses step.progress (not stream.chunk) so the event flows
            through LegacyProgressSink → on_progress →
            ProgressDisplay.update_message() for real-time inline
            display in the Rich spinner (e.g. "thinking: ...").
            """
            try:
                emitter.emit_sync(WorkflowEvent.step_progress(
                    run_id=run_id,
                    workflow=workflow_name,
                    step=step_name,
                    message=thinking_text,
                    data={"kind": "thinking"},
                ))
            except Exception:
                pass

        def on_stream_end() -> None:
            try:
                emitter.emit_sync(
                    WorkflowEvent.stream_end(run_id, workflow_name, step_name),
                )
            except Exception:
                pass

        # Use async streaming (no more run_in_executor!)
        wiki_result = await template.generate_wiki_stream(
            ctx.data["md_content"],
            kb._build_filemeta(
                ctx.data["file_id"], ctx.data["file_type"],
                source_path.name, ctx.data["file_hash"],
                source_path.stat().st_size, str(source_path),
                ctx.data["raw_dest"], Path(),
                ctx.data["md_content"], "", True, True, True,
                ctx.data.get("cat_name", ""),
                ctx.data.get("tags") or [], ctx.data.get("all_keys", []),
            ),
            model_config=llm_config,
            on_chunk=on_chunk,
            on_thinking=on_thinking,
            on_stream_end=on_stream_end,
        )

        wiki_content = wiki_result.content
        tokens_input = wiki_result.tokens_input
        tokens_output = wiki_result.tokens_output
        tokens_total = wiki_result.total_tokens

        # Emit token metrics
        emitter.emit_sync(WorkflowEvent(
            event_type="metric.token",
            run_id=run_id,
            workflow_name=workflow_name,
            step_name=step_name,
            message=f"Tokens: {tokens_input} in / {tokens_output} out",
            data={
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "tokens_total": tokens_total,
                "requests": wiki_result.requests,
            },
        ))

        if wiki_content.startswith("[LLM"):
            # Log the actual error message so it's visible in logs/terminal
            import logging
            logging.getLogger("mnemo").error(
                "Wiki generation failed for '%s': %s",
                source_path.name, wiki_content,
            )
            return {
                "wiki_content": "",
                "wiki_info": {
                    "status": "failed", "generated_at": "",
                    "template": template.name,
                    "model": llm_config.get("model", ""), "chars": 0,
                    "tokens_used": tokens_total,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "error": wiki_content,
                },
            }

        wiki_path = (
            kb.data_dir / "raw_wiki" / ctx.data["cat_name"]
            / ctx.data["file_type"] / ctx.data["chunk_dir"]
            / f"{source_path.stem}.md"
        )
        wiki_path.parent.mkdir(parents=True, exist_ok=True)
        wiki_path.write_text(wiki_content, encoding="utf-8")

        # -- Diagnostic ----------------------------------------------------
        if ctx.diagnostic and ctx.diagnostic.enabled:
            max_preview = ctx.diagnostic.max_text_preview_chars
            ctx.emit("step.progress", step_name="generate_wiki",
                     data={"_diagnostic": {
                         "template": template.name,
                         "model": llm_config.get("model", ""),
                         "base_url": llm_config.get("base_url", ""),
                         "temperature": llm_config.get("temperature", ""),
                         "max_tokens": llm_config.get("max_tokens", ""),
                         "tokens_input": tokens_input,
                         "tokens_output": tokens_output,
                         "tokens_total": tokens_total,
                         "wiki_preview": wiki_content[:max_preview],
                         "wiki_chars": len(wiki_content),
                     }})

        return {
            "wiki_content": wiki_content,
            "wiki_info": {
                "status": "done",
                "generated_at": datetime.now(UTC).isoformat(),
                "template": template.name,
                "model": llm_config.get("model", ""),
                "chars": len(wiki_content),
                "tokens_used": tokens_total,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
            },
        }
    except Exception as exc:
        import logging
        logging.getLogger("mnemo").error(
            "Wiki generation exception for '%s': %s",
            source_path.name, exc, exc_info=True,
        )
        return {
            "wiki_content": "",
            "wiki_info": {
                "status": "failed", "generated_at": "",
                "template": getattr(template, 'name', ''),
                "model": llm_config.get("model", ""), "chars": 0,
                "tokens_used": 0, "tokens_input": 0, "tokens_output": 0,
                "error": str(exc),
            },
        }


# ============================================================================
# Step 6: Entity Extraction
# ============================================================================


@StepRegistry.register_function("extract_entities")
async def extract_entities(ctx: WorkflowContext) -> dict:
    """Extract entities and relations from markdown into graph store."""
    kb = _get_kb(ctx)
    md_content = ctx.data.get("md_content", "")
    if not md_content:
        return {}

    if "graph_entities" not in getattr(kb, '_required_caps', set()):
        return {}

    from mnemo.core.interfaces.entity_extractor import IEntityExtractor
    from mnemo.core.param_config import get_config as _get_cfg

    ee_cfg = _get_cfg(IEntityExtractor)
    if "entity_extraction" in kb.config:
        ee_cfg = {**ee_cfg, **kb.config["entity_extraction"]}

    if not ee_cfg.get("enabled_on_add", True):
        return {}

    try:
        entities, relations = kb.entity_extractor.extract(md_content)
        if entities:
            entity_ids = kb.graph_store.upsert_entities(entities)
            relevance = [1.0] * len(entity_ids)
            kb.graph_store.link_file_entities(
                ctx.data["file_id"], entity_ids, relevance,
            )
        if relations:
            kb.graph_store.add_relations(relations)
    except Exception:
        import logging
        logging.getLogger("mnemo").warning(
            "Entity extraction failed for '%s'", ctx.data["file_id"],
        )

    return {"graph_entities": "done"}


# ============================================================================
# Step 7: Embed
# ============================================================================


@StepRegistry.register_function("embed_chunks")
async def embed_chunks(ctx: WorkflowContext) -> dict[str, Any]:
    """Generate embeddings for markdown chunks and store in vector DB."""
    kb = _get_kb(ctx)
    embed_info = {"raw": "skipped", "md": "pending",
                   "wiki": "pending", "metadata": "pending"}

    if not ctx.data.get("auto_embed") or "embeddings" not in kb._required_caps:
        return {"embed_info": embed_info}

    md_content = ctx.data.get("md_content", "")
    if not md_content:
        return {"embed_info": embed_info}

    try:
        chunker, chunker_cfg = kb._resolve_chunker(ctx.data["file_type"])
        chunks = chunker.chunk(md_content, chunker_cfg)

        # -- Diagnostic: chunker stage --------------------------------------
        if ctx.diagnostic and ctx.diagnostic.enabled:
            max_preview = ctx.diagnostic.max_text_preview_chars
            chunk_previews = []
            for c in chunks[:10]:  # cap at 10 chunk previews
                chunk_previews.append({
                    "index": c.chunk_index,
                    "text_preview": c.text[:max_preview],
                    "text_total_chars": len(c.text),
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "section_header": c.metadata.get("section_header", ""),
                    "parent_id": c.metadata.get("parent_id", ""),
                })
            ctx.emit("step.progress", step_name="embed_chunks",
                     data={"_diagnostic": {
                         "substage": "chunker",
                         "chunker": chunker.name if hasattr(chunker, 'name') else type(chunker).__name__,
                         "chunker_config": {k: str(v) for k, v in chunker_cfg.items()},
                         "chunk_count": len(chunks),
                         "chunk_previews": chunk_previews,
                     }})

        from mnemo.core.embedder import get_embedder, get_dimension, get_model_name
        model_name = get_model_name()
        chunk_texts = [c.text for c in chunks]
        try:
            result = await get_embedder().embed_documents(chunk_texts)
            vectors = [list(v) for v in result.embeddings]
        except Exception:
            vectors = [[0.0] * get_dimension() for _ in chunk_texts]

        # -- Diagnostic: embedding stage -----------------------------------
        if ctx.diagnostic and ctx.diagnostic.enabled:
            from mnemo.core.diagnostics import truncate_vector
            max_dims = ctx.diagnostic.max_vector_preview_dims
            vector_previews = [
                truncate_vector(v, max_dims) for v in vectors[:3]
            ]  # cap at 3 vector previews
            ctx.emit("step.progress", step_name="embed_chunks",
                     data={"_diagnostic": {
                         "substage": "embedding",
                         "model": model_name,
                         "dimension": get_dimension(),
                         "vector_count": len(vectors),
                         "vector_previews": vector_previews,
                         "table": "raw_md",
                     }})

        kb.vector_store.add_vectors(
            "raw_md",
            ids=[ctx.data["file_id"]] * len(vectors),
            vectors=vectors,
            metadata={
                "file_type": [ctx.data["file_type"]] * len(vectors),
                "chunk_index": [c.chunk_index for c in chunks],
                "model": [model_name] * len(vectors),
                "start_char": [c.start_char for c in chunks],
                "end_char": [c.end_char for c in chunks],
                "section_header": [
                    c.metadata.get("section_header", "") for c in chunks
                ],
                "parent_id": [
                    c.metadata.get("parent_id", "") for c in chunks
                ],
            },
        )
        embed_info["md"] = "done"

        # Also embed wiki content if available
        wiki_content = ctx.data.get("wiki_content", "")
        if wiki_content:
            try:
                wiki_chunks = chunker.chunk(wiki_content, chunker_cfg)
                wiki_texts = [c.text for c in wiki_chunks]
                try:
                    wiki_result = await get_embedder().embed_documents(wiki_texts)
                    wiki_vectors = [list(v) for v in wiki_result.embeddings]
                except Exception:
                    wiki_vectors = [[0.0] * get_dimension() for _ in wiki_texts]
                kb.vector_store.add_vectors(
                    "raw_wiki",
                    ids=[ctx.data["file_id"]] * len(wiki_vectors),
                    vectors=wiki_vectors,
                    metadata={
                        "file_type": [ctx.data["file_type"]] * len(wiki_vectors),
                        "chunk_index": [c.chunk_index for c in wiki_chunks],
                        "model": [model_name] * len(wiki_vectors),
                        "start_char": [c.start_char for c in wiki_chunks],
                        "end_char": [c.end_char for c in wiki_chunks],
                        "section_header": [
                            c.metadata.get("section_header", "")
                            for c in wiki_chunks
                        ],
                        "parent_id": [
                            c.metadata.get("parent_id", "")
                            for c in wiki_chunks
                        ],
                    },
                )
                embed_info["wiki"] = "done"
            except Exception as exc:
                import logging
                logging.getLogger("mnemo").warning(
                    "Wiki embedding failed for '%s': %s",
                    ctx.data["file_id"], exc,
                )
                embed_info["wiki"] = "failed"
        else:
            embed_info["wiki"] = "skipped"

    except Exception as exc:
        import logging
        logging.getLogger("mnemo").warning(
            "Embedding failed for '%s': %s", ctx.data["file_id"], exc,
        )
        embed_info["md"] = "failed"
        embed_info["wiki"] = "failed"

    return {"embed_info": embed_info}


# ============================================================================
# Step 8: Index
# ============================================================================


@StepRegistry.register_function("write_index")
async def write_index(ctx: WorkflowContext) -> dict[str, Any]:
    """Write metadata file, insert into SQLite indexer, register keys."""
    kb = _get_kb(ctx)
    source_path = ctx.get_input("source_path")
    file_id = ctx.data["file_id"]
    all_keys = ctx.data.get("all_keys", [])
    tags = ctx.data.get("tags") or []
    note = ctx.config.get("note", "")
    auto_md = ctx.data.get("auto_md", False)
    auto_wiki = ctx.data.get("auto_wiki", False)
    auto_embed = ctx.data.get("auto_embed", False)
    wiki_info = ctx.data.get("wiki_info")
    embed_info = ctx.data.get("embed_info", {})

    from mnemo.core.metadata_writer import MetadataWriter
    writer = MetadataWriter(kb.data_dir)

    meta_dict = {
        "file_type": ctx.data["file_type"],
        "filename": source_path.name,
        "file_hash": ctx.data["file_hash"],
        "file_size": source_path.stat().st_size,
        "source_path": str(source_path),
        "category": ctx.data.get("cat_name", ""),
        "tags": tags,
        "keys": all_keys,
        "keywords": [],
        "added_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "auto_md": auto_md,
        "auto_wiki": auto_wiki,
        "auto_embed": auto_embed,
        "parser_name": ctx.data.get("parser").name if ctx.data.get("parser") else "",
        "template_name": ctx.data.get("template").name if ctx.data.get("template") else "",
        "note": note,
        "chunk": ctx.data.get("chunk_dir", ""),
        "embed_raw": embed_info.get("raw", "skipped"),
        "embed_md": embed_info.get("md", "pending"),
        "embed_wiki": embed_info.get("wiki", "pending"),
        "embed_metadata": embed_info.get("metadata", "pending"),
        "custom": {"note": note} if note else {},
    }

    md_info = ctx.data.get("md_info")
    meta_path = writer.write(file_id, meta_dict, md_info=md_info, wiki_info=wiki_info)

    internal_meta = kb._build_filemeta(
        file_id, ctx.data["file_type"], source_path.name,
        ctx.data["file_hash"], source_path.stat().st_size,
        str(source_path), ctx.data["raw_dest"], meta_path,
        ctx.data.get("md_content", ""), ctx.data.get("wiki_content", ""),
        auto_md, auto_wiki, auto_embed,
        ctx.data.get("cat_name", ""), tags, all_keys,
    )
    kb.indexer.insert_file(internal_meta)

    if auto_wiki and wiki_info and wiki_info.get("status") == "done":
        kb.indexer.update_file(file_id, wiki_status="done")
    if auto_embed and embed_info.get("md") == "done":
        kb.indexer.update_file(file_id, embed_status="done")

    for k in all_keys:
        kb.key_manager.register_key(k)
    kb.key_manager.add_file_keys(file_id, all_keys)

    _wiki_status = FileStatus.SKIPPED
    if auto_wiki:
        if wiki_info and wiki_info.get("status") == "done":
            _wiki_status = FileStatus.DONE
        elif wiki_info and wiki_info.get("status") == "failed":
            _wiki_status = FileStatus.FAILED
        else:
            _wiki_status = FileStatus.PENDING

    # Build processing_detail for CLI display
    from mnemo.core.embedder import get_model_name as _get_emb_model, get_dimension
    _processing_detail: dict[str, Any] = {}
    if wiki_info:
        _processing_detail["wiki"] = wiki_info
    if embed_info:
        _processing_detail["embedding"] = {
            "model": _get_emb_model(),
            "dimension": get_dimension(),
            "chunks": embed_info.get("md_chunks", 0),
        }

    result = FileInfo(
        id=file_id,
        file_type=ctx.data["file_type"],
        filename=source_path.name,
        file_size=source_path.stat().st_size,
        file_hash=ctx.data["file_hash"],
        raw_path=str(ctx.data["raw_dest"].relative_to(kb.data_dir)),
        metadata_path=str(meta_path.relative_to(kb.data_dir)),
        source_path=str(source_path),
        category=ctx.data.get("cat_name", ""),
        tags=tags,
        keys=all_keys,
        added_at=meta_dict["added_at"],
        updated_at=meta_dict["updated_at"],
        md_status=FileStatus.DONE if auto_md else FileStatus.SKIPPED,
        wiki_status=_wiki_status,
        embed_status=(
            FileStatus.DONE if (auto_embed and embed_info.get("md") == "done")
            else FileStatus.FAILED if (auto_embed and embed_info.get("md") == "failed")
            else FileStatus.SKIPPED
        ),
        processing_detail=_processing_detail,
    )

    # -- Diagnostic: final summary -------------------------------------------
    if ctx.diagnostic and ctx.diagnostic.enabled:
        from mnemo.core.embedder import get_model_name as _emb_model, get_dimension
        ctx.emit("step.progress", step_name="write_index",
                 data={"_diagnostic": {
                     "file_id": file_id,
                     "file_type": ctx.data["file_type"],
                     "category": ctx.data.get("cat_name", ""),
                     "parser": ctx.data.get("parser").name if ctx.data.get("parser") else "",
                     "template": ctx.data.get("template").name if ctx.data.get("template") else "",
                     "auto_md": auto_md,
                     "auto_wiki": auto_wiki,
                     "auto_embed": auto_embed,
                     "md_status": _processing_detail.get("md", {}).get("status", ""),
                     "wiki_status": (_wiki_status.value
                                     if hasattr(_wiki_status, 'value') else str(_wiki_status)),
                     "embed_status": (embed_info.get("md", "pending")),
                     "embed_model": _emb_model(),
                     "embed_dimension": get_dimension(),
                     "keys": all_keys,
                     "tags": tags,
                 }})

    return {"file_info": result}
