# Mnemo (忆枢)

> AI-agent-friendly personal knowledge base — External Memory Hub
> 面向 AI Agent 的个人知识库 — 外置记忆中枢

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mnemo is a file-based personal knowledge base. Ingest files → auto-classify → Markdown → LLM wiki → embed → hybrid search (vector ANN + BM25 keyword + graph RRF fusion).

Mnemo 是一个文件化的个人知识库。摄入文件 → 自动二级分类 → 转 Markdown → LLM Wiki 摘要 → 向量嵌入 → 混合检索（向量 ANN + BM25 关键词 + 图谱 RRF 融合）。

## Design Philosophy / 设计理念

- **Files ARE the database** — unified directory structure. cp/tar/rsync = migrate/backup/merge.
  **文件即数据库** — 统一目录结构，复制即迁移，tar 即备份。
- **AI-agent-friendly** — CLI, Python API, REST API, MCP server. Directly callable by Claude, Codex, etc.
  **AI Agent 友好** — 四接口统一（CLI / Python / REST / MCP），可被 Agent 直接调用。
- **Fully pluggable** — every component is a plugin interface. `PluginHub.get(IXxx, "name")` for lookup. `__init_subclass__` auto-registration — no manual decorators.
  **全插件化** — 每个组件都是插件接口，`__init_subclass__` 自动注册，无需手动装饰器。
- **Category + Type** — broad category (docs/data/code/img/...) → specific type (pdf/csv/py/...). Parser/template resolved via fallback chain.
  **二级分类** — Category（大类）→ Type（具体类型），Parser/Template 按回退链自动匹配。

## Quick Start / 快速开始

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone
git clone https://github.com/YOUR_USER/mnemo.git
cd mnemo
uv sync

# Initialize a knowledge base
uv run mnemo init ~/my-knowledge

# Add a file
uv run mnemo add --file paper.pdf --keys "research::nlp"

# Search
uv run mnemo search "transformer architecture" --mode hybrid

# Start REST API server
uv run mnemo serve

# Python API
from mnemo.api import MnemoAPI

with MnemoAPI("~/my-knowledge") as api:
    api.add("paper.pdf", keys=["research::nlp"])
    results = api.search("attention mechanism")
    for r in results:
        print(f"[{r.score:.2f}] {r.snippet}")
```

## Architecture / 架构

```
MCP / REST / CLI / Python API
        │
   MnemoAPI (facade)
        │
   KnowledgeBase (core)
        │
   ├── PluginHub           ← unified plugin registry (auto __init_subclass__)
   ├── WorkflowEngine       ← DAG engine (graphlib + asyncio)
   ├── AgentRegistry        ← pydantic-ai Agent factory + tools
   ├── EventBus             ← unified observability (CLI / Log / Metrics sinks)
   └── Diagnostics          ← pipeline trace (--diagnose flag, JSONL output)
```

### Pipeline / 管道

| Workflow | Steps |
|----------|-------|
| **add** | validate → copy → metadata → parse_to_md → generate_wiki → extract_entities → embed_chunks → write_index |
| **search** | query_embedding → vector_ann + keyword_bm25 + graph_expand → RRF fuse → rerank → filter |
| **ask** (RAG) | analyze → search → assemble → generate → verify |

### Plugins / 插件 (40+ implementations)

| Interface | Implementations |
|-----------|----------------|
| IParser | text, code, pdf, image, url, csv, tsv, json, xml, npy, hdf5 |
| ITemplate | note, paper, dataset, code |
| IChunker | langchain (default), paragraph, token, fixed_size, semantic, small_to_big |
| ISearcher | lightrag (default, vector+BM25+graph RRF), keyword (BM25+jieba), simple (grep) |
| IVectorStore | lancedb (IVF_PQ ANN) |
| IGraphStore | sqlite (entity/relation graph) |
| IIndexer / IKeyManager | sqlite (CRUD + JSON1 tags + recursive CTE) |

### Diagnostics / 诊断

```bash
# Capture full pipeline trace for search quality optimization
uv run mnemo add --diagnose paper.pdf
uv run mnemo search --diagnose "query" --verbose

# Inspect traces
jq 'select(.stage=="embed_chunks")' ~/my-knowledge/.mnemo/diagnostics/add_*.jsonl
jq 'select(.stage=="vector_ann") | .data.distance_stats' ~/my-knowledge/.mnemo/diagnostics/search_*.jsonl
```

Each trace captures: chunk previews, embedding model/dimension, vector distances (min/max/mean/std), per-channel RRF scores, and stage-level timing for every pipeline stage.

## Project Structure / 项目结构

```
src/mnemo/
├── api/                    # Public API — MnemoAPI + MCP + REST
│   ├── client.py           #   MnemoAPI facade
│   ├── server.py           #   FastAPI REST server (/docs)
│   ├── mcp_server.py       #   MCP stdio server
│   └── types.py            #   Public types (FileInfo, SearchResult, ...)
├── core/
│   ├── plugin_base.py      #   PluginBase + PluginHub (unified plugin system)
│   ├── diagnostics.py      #   Pipeline diagnostic tracing
│   ├── kb.py               #   KnowledgeBase core
│   ├── interfaces/         #   15 plugin interfaces (IParser, IChunker, ISearcher, ...)
│   ├── workflow/           #   WorkflowEngine DAG (graphlib + asyncio)
│   │   ├── engine.py       #     Async parallel execution
│   │   ├── events.py       #     EventBus (Emitter + sinks)
│   │   ├── context.py      #     WorkflowContext + Deps
│   │   ├── add_steps.py    #     File ingestion steps
│   │   ├── search_steps.py #     Search pipeline steps
│   │   ├── agent_registry.py #   Agent + Tool factory
│   │   └── tools.py        #     Built-in tools (search_kb, get_context, ...)
│   ├── agent_manager.py    #   pydantic-ai Agent factory
│   ├── embedder.py         #   pydantic-ai Embedder singleton
│   └── param_config.py     #   Parameter config (MRO merge, env_var)
├── plugins/
│   ├── parsers/            #   text, code, pdf, data_file, image, url
│   ├── templates/          #   note, paper, dataset, code
│   ├── chunkers/           #   langchain, paragraph, token, fixed_size, semantic, small_to_big
│   ├── searchers/          #   lightrag, keyword, simple
│   ├── indexers/           #   sqlite
│   ├── key_managers/       #   sqlite
│   ├── vector_stores/      #   lancedb
│   ├── graph_stores/       #   sqlite
│   ├── entity_extractors/  #   llm
│   ├── file_categories/    #   docs, data, code, code.py, img, audio, video, web, other
│   ├── importers/          #   tar
│   ├── exporters/          #   tar
│   └── syncers/            #   rclone
├── builtin/
│   └── workflows/          #   Built-in workflow TOML definitions
├── cli/                    #   Click CLI (20 commands)
└── utils/                  #   config, logging, hash, path safety
```

## Plugin Development / 插件开发

Define an interface:

```python
from abc import ABC, abstractmethod
from typing import ClassVar
from mnemo.core.plugin_base import PluginBase

class IMyPlugin(PluginBase, ABC):
    __plugin_interface__ = True
    name: ClassVar[str] = "my_plugin"
    plugin_path: ClassVar[str] = "my_plugins"

    @abstractmethod
    def do_something(self, text: str) -> str: ...
```

Implement it — no decorator needed, auto-registered:

```python
class MyPlugin(IMyPlugin):
    __plugin_impl__ = True
    name = "default"

    def do_something(self, text: str) -> str:
        return text.upper()

# Lookup — returns IMyPlugin, not Any
plugin = PluginHub.get(IMyPlugin, "default")
```

## Development / 开发

```bash
uv sync --all-extras      # install all dependencies
uv run pytest             # run tests (221 pass)
uv run mypy src/mnemo/    # type check
uv run ruff check .       # lint
```

See [AGENTS.md](AGENTS.md) for full development guide.

## License / 许可证

MIT — see [LICENSE](LICENSE)
