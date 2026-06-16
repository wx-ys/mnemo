# Mnemo — AI Agent 开发指南

**文件化的个人知识库，AI Agent 的外置记忆中枢。**

## 行为准则

1. **先想再写** — 不确定就问，不假设。有更简单的方案就说。
2. **简洁优先** — 不写投机性代码，不为单一调用建抽象层。
3. **改动** — 匹配已有风格, 先改接口再改实现, 先改核心再改边缘。
4. **可验证执行** — 声明目标 + 验证方式，写测试循环迭代直到通过。

## 开发环境

### 前置条件

- **Python 3.11+** (当前开发使用 3.14)
- **[uv](https://docs.astral.sh/uv/)** — Python 包管理器，替代 pip/venv/pip-tools
- **[git](https://git-scm.com/)** — 版本控制

### 初始设置

```bash
git clone https://github.com/YOUR_USER/mnemo.git
cd mnemo

# uv 自动创建 .venv 并安装所有依赖
uv sync
uv sync --all-extras  # 安装全部可选依赖 (parsers, llm, vector, search, embedding)

# 验证
uv run mnemo --version
uv run pytest
```

### 日常开发循环

```bash
# 1. 创建分支
git checkout -b feat/my-feature

# 2. 开发 + 测试
uv run pytest tests/test_xxx.py -x -q

# 3. 类型检查
uv run mypy src/mnemo/

# 4. Lint
uv run ruff check src/mnemo/ tests/

# 5. 提交 (遵循 conventional commits)
git add -A
git commit -m "feat: add X" -m "详细说明"
git push origin feat/my-feature

# 6. 创建 PR (或直接合并到 main)
gh pr create --title "feat: add X" --body "..."
```

### 一键检查

```bash
# 提交前运行全部检查
uv run ruff check src/mnemo/ tests/ && \
  uv run mypy src/mnemo/ && \
  uv run pytest -q
```

### 依赖管理

```bash
uv add <package>           # 添加运行时依赖
uv add --dev <package>     # 添加开发依赖
uv add --optional <grp> <pkg>  # 添加可选依赖组

# 依赖组: dev, search, vector, embedding, parsers, llm, all
# 参见 pyproject.toml [project.optional-dependencies] 和 [dependency-groups]
```

### 常用命令

| 命令 | 用途 |
|------|------|
| `uv sync` | 同步依赖到 .venv |
| `uv run pytest -q` | 运行测试 (221 pass) |
| `uv run mypy src/mnemo/` | 类型检查 |
| `uv run ruff check .` | Lint 检查 |
| `uv run mnemo --help` | CLI 帮助 |

## 架构

```
MCP / REST / CLI / Python API
        │
   MnemoAPI (facade)         ← 统一对外门面 (api/client.py)
        │
   KnowledgeBase             ← 核心逻辑 (core/kb.py)
   │  │
   │  ├── Workflow Engine    ← DAG 工作流引擎 (core/workflow/)
   │  │   ├── WorkflowDAG     ← graphlib DAG (TOML → 并行执行)
   │  │   ├── WorkflowEngine  ← 异步并行 (retry/timeout/condition)
   │  │   ├── add_steps.py    ← 文件摄入 8 步骤 (validate → index)
   │  │   ├── search_steps.py ← 检索步骤 (plan → search → fuse → rerank)
   │  │   ├── StepRegistry    ← 函数/Agent/子工作流注册
   │  │   ├── EventBus        ← 统一可观测 (CLI/Log/Metrics sinks)
   │  │   ├── AgentRegistry   ← pydantic-ai Agent + Tool 工厂
   │  │   └── ToolLibrary     ← 内置工具 (search_kb, get_context, ...)
   │  │
   │  └── AskPipeline         ← RAG 问答 (core/kb_ask.py)
   │
   PluginHub  ──→  Interface (ABC)  ──→  具体实现 (plugins/)
   AgentManager              ← pydantic-ai Agent 工厂 (core/agent_manager.py)
   embedder.py               ← pydantic-ai Embedder 单例 (core/embedder.py)
   diagnostics.py            ← 管道诊断追踪 (core/diagnostics.py)
```

所有业务逻辑通过 **WorkflowEngine** 执行 — 文件摄入 (add)、智能检索 (search)、
RAG 问答 (ask) 均为 DAG 驱动的 FunctionStep/AgentStep 编排。

依赖方向不可逆。所有组件通过 PluginHub 获取，不直接 import 具体实现。
LLM Agent 通过 `AgentManager` / `AgentRegistry`，Embedding 通过 `get_embedder()` — 均不经过 PluginHub。

### 多接口统一

| 接口 | 入口 | 协议 |
|------|------|------|
| **Python API** | `from mnemo.api import MnemoAPI` | 直接调用 |
| **CLI** | `mnemo <command>` | Click (rich-click) |
| **REST API** | `mnemo serve` → `http://localhost:8765` | FastAPI + OpenAPI |
| **MCP** | `mnemo mcp` | Model Context Protocol (stdio) |

### Category + Type 二级分类

```
docs:  pdf, docx, ppt, txt, md, rst, log, tex     code:  py, js, rs, go, java, c, cpp ...
data:  csv, npy, hdf5, json                        img:   jpg, png, svg, gif, webp, bmp
audio: mp3, wav, flac                              video: mp4, mkv
web:   html, url                                   other: 未识别 (fallback)
```

Parser 回退: type级 → category级 → `"text"`。Template 回退: type级 → category级 → `"note"`。

### PluginHub — 统一插件注册中心

**1 个 PluginHub** 替代 15 个独立 Registry 单例。通过 `__init_subclass__` 钩子实现自动注册，无需手动装饰器：

```python
from mnemo.core.plugin_base import PluginBase, PluginHub

# 定义接口 — __plugin_interface__ = True + ABC
class IIndexer(PluginBase, ABC):
    __plugin_interface__ = True
    name: ClassVar[str] = "indexer"
    plugin_path: ClassVar[str] = "indexers"

# 定义实现 — __plugin_impl__ = True 自动注册
class SQLiteIndexer(IIndexer):
    __plugin_impl__ = True
    name = "sqlite"

# 类型安全的查找 — 返回 IIndexer (不是 Any)
indexer = PluginHub.get(IIndexer, "sqlite")  # 懒创建
```

**两个标记**: `__plugin_interface__ = True` (接口) / `__plugin_impl__ = True` (实现)。
中间基类不加标记，不被注册。重复注册 (同名同接口) 在类定义时立即抛出 TypeError。
`PluginHub.get(Iface, "name")` 返回精确类型 `Iface`，IDE 自动补全。

### AgentManager — pydantic-ai Agent 工厂

LLM 不使用 PluginHub 插件模式，通过 config-driven singleton 管理：

```python
from mnemo.core.agent_manager import AgentManager

# 直接返回 pydantic_ai.Agent 实例
agent = AgentManager.get_instance().get_agent("default")
result = agent.run_sync("What is AI?")

# 结构化输出:
agent = AgentManager.get_instance().get_agent("default", output_type=WikiOutput)
```

多 agent 配置通过 `config.toml` 的 `[agent.xxx]` 段定义。

### AgentRegistry — v2 Tool 增强 (core/workflow/agent_registry.py)

`AgentRegistry` 扩展 `AgentManager`，增加 ToolLibrary 工具解析和结构化输出支持：

```python
from mnemo.core.workflow import AgentRegistry

# 带内置工具的 Agent:
agent = AgentRegistry._get().get_agent_with_tools(
    "default",
    tools=["search_kb", "get_file_context"],
    output_type=AskOutput,
)

# Tool 注册 (pydantic-ai 工具):
from mnemo.core.workflow import ToolLibrary

@ToolLibrary.register("my_tool")
def my_tool(query: str) -> str: ...
```

内置工具: `search_kb`, `get_file_context`, `chunk_text`, `resolve_file_ref`。

### Workflow Engine — DAG 工作流引擎 (core/workflow/)

新一代工作流系统，基于 `graphlib.TopologicalSorter` (Python stdlib) 实现声明式 DAG 编排：

```python
from mnemo.core.workflow import (
    WorkflowDAG, WorkflowEngine, WorkflowContext,
    StepConfig, StepRegistry, FunctionStep, AgentStep,
)

# TOML → DAG → 并行执行
from mnemo.core.workflow.config import WorkflowConfigLoader
loader = WorkflowConfigLoader()
config = loader.load("add")  # 加载 add.workflow.toml

dag = WorkflowDAG.from_config(config)  # graphlib DAG
ctx = WorkflowContext(workflow_name="add", kb=kb)

engine = WorkflowEngine()
result = await engine.execute(dag, ctx)  # 异步并行执行

# KB 入口 (v2):
kb.add("file.pdf", use_new_workflow=True)
```

**关键概念**:
- **Step**: 原子执行单元 — `FunctionStep`(确定函数) / `AgentStep`(LLM) / `PipelineStep`(嵌套)
- **DAG**: TOML `depends_on` → graphlib 拓扑排序 → 自动并行化独立分支
- **WorkflowContext**: 统一数据流 + pydantic-ai `Deps` 注入容器
- **条件执行**: 每个 step 可选 Jinja2 `condition` 表达式
- **EventBus**: `EventEmitter` → `CLISink`(Rich) / `LogSink`(JSON) / `MetricsSink`

**工作流 TOML 示例**:
```toml
[workflow.steps.judge]
type = "agent"
agent_name = "default"
tools = ["search_kb"]
depends_on = ["copy"]
condition = "ctx.data.config.auto_wiki"
timeout_seconds = 30
retry = 2
```

工作流定义位于 `src/mnemo/builtin/workflows/`，用户可通过 `{data_dir}/.mnemo/workflows/` 覆盖。

### Embedder — 模块级单例

一个知识库使用一个 embedding 模型，通过 `core/embedder.py` 提供统一的 `get_embedder()` 入口：

```python
from mnemo.core.embedder import get_embedder, init_embedder

# KB.__init__ 中初始化一次:
init_embedder(config)

# 库内任意位置直接获取:
embedder = get_embedder()
result = embedder.embed_documents_sync(["text1", "text2"])
vectors = [list(v) for v in result.embeddings]

# query embedding:
result = get_embedder().embed_query_sync("search query")
query_vec = list(result.embeddings[0])
```

配置通过 `config.toml` 的 `[embedder]` 单段定义。

### config_schema — MRO 继承

每个 ABC 和插件通过 `config_schema` 声明参数，使用 `Param` dataclass:

```python
from mnemo.core.interfaces.param_spec import Param

config_schema: dict[str, Param] = {
    "model":   Param(type="str", default="gpt-4o", desc="Model name"),
    "api_key": Param(type="str", env_var="LLM_API_KEY",
                     desc="API key (reads from LLM_API_KEY env var)"),
    "max_tokens": Param(type="int", default=2048),
}
```

- `env_var` — 环境变量名，解析时 `os.environ.get(env_var)` 读取
- 解析优先级: `MNEMO_*` 覆盖 > TOML 显式值 > `env_var` 环境变量 > `Param.default`
- MRO 自动合并继承，子类**只声明自己新增或覆盖的字段**
- **⚠️ 禁止 `**Parent.config_schema` 展开** — MRO 已处理继承，手动展开导致 `_get_own_schema()` 误判、TOML 重复、父类变更不传递
- 接口级 `default_plugin` 选择活跃实现。文件分类配置写入 `file_categories.toml`，层级回退（`code.py` → `code` → `other`）

---

## Chunking 系统 (IChunker)

6 个内置 chunker。优先级: `file_categories.toml` > `config.toml` chunker.default_plugin > `"default"`。

| Chunker | name | 策略 |
|---------|------|------|
| **LangChainChunker** | `"default"` | 感知文件类型：code→语言分隔符, md→Markdown header, html→HTML header, fallback→递归字符分割 |
| ParagraphChunker | `"paragraph"` | `\n\n` 段落边界累积 |
| TokenChunker | `"token"` | tiktoken token 计数 |
| FixedSizeChunker | `"fixed_size"` | 严格字符切片 |
| SemanticChunker | `"semantic"` | langchain-experimental 嵌入相似度检测 (需 embedder 注入) |
| SmallToBigChunker | `"small_to_big"` | 父子两级：parent 2000 chars → `raw_md_parents` 表 (纯文本)，child 500 chars → embed → `raw_md` 表 |

`ChunkInfo`: `text`, `chunk_index`, `start_char`, `end_char`, `metadata` (section_header, parent_id, chunk_level, language...)。Small-to-Big 检索时 child 精确匹配，parent 提供扩展上下文。

---

## Searcher 系统 (ISearcher)

3 个实现。每个 Searcher 声明 `required_capabilities` (`embeddings`, `graph_entities`, `markdown_content`)，`add()` 据此自动裁剪 ingestion 管道。

| Searcher | name | 依赖 | 策略 |
|----------|------|------|------|
| **LightRAGSearcher** | `"default"` | Embedder(KB注入)+VectorStore+GraphStore(可选) | 向量ANN + BM25 + 图谱 RRF 三路融合 |
| KeywordSearcher | `"keyword"` | jieba | BM25 关键词 |
| SimpleSearcher | `"simple"` | 无 | grep 子字符串 |

LightRAGSearcher 0-arg 构造，Embedder 由 KB 注入，其他依赖从 PluginHub 自解析。

---

## 项目结构

```
src/mnemo/
├── api/                    # 对外接口层 — MnemoAPI + MCP + REST
│   ├── client.py           #   MnemoAPI 统一门面
│   ├── server.py           #   FastAPI REST server
│   ├── mcp_server.py       #   MCP stdio server (5 tools)
│   ├── types.py            #   公开类型 (FileInfo, SearchResult, ...)
│   └── ask_types.py        #   AskResponse, Citation
├── core/
│   ├── workflow/           # 【NEW v2】工作流引擎
│   │   ├── events.py       #   EventBus (Emitter + Null/Log/Metrics/CLI sinks)
│   │   ├── context.py      #   WorkflowContext + WorkflowDeps (pydantic-ai DI)
│   │   ├── step.py         #   Step ABC + FunctionStep + AgentStep + PipelineStep
│   │   ├── dag.py          #   WorkflowDAG (封装 graphlib.TopologicalSorter)
│   │   ├── engine.py       #   WorkflowEngine (asyncio.gather 并行执行)
│   │   ├── config.py       #   WorkflowConfigLoader (TOML 加载 + 三层合并)
│   │   ├── agent_registry.py # AgentRegistry (AgentManager 升级 + Tool 支持)
│   │   ├── tools.py        #   ToolLibrary (4 内置 tools)
│   │   ├── compat.py       #   v1 PipelineStage → v2 FunctionStep 兼容层
│   │   └── search_steps.py #   搜索/Ask 步骤实现
│   ├── interfaces/         # 15 ABCs (IParser, IChunker, ISearcher 等)
│   ├── plugin_base.py      # PluginBase + PluginHub (统一插件注册)
│   ├── diagnostics.py      # DiagnosticContext + DiagnosticSink (管道诊断追踪)
│   ├── kb.py               # KnowledgeBase
│   ├── kb_ask.py           # AskPipeline (RAG 问答) [稳定]
│   ├── pipeline.py         # PipelineStage + AddPipeline (摄入编排) [稳定]
│   ├── param_config.py     # 参数配置 (MRO合并, env_var, TOML生成)
│   ├── agent_manager.py    # pydantic-ai Agent 工厂 [委托到 AgentRegistry]
│   ├── embedder.py         # pydantic-ai Embedder 单例 (get_embedder)
│   └── prompt_manager.py   # 提示词管理
├── builtin/
│   └── workflows/          # 【NEW】内置工作流 TOML 定义
│       ├── add.workflow.toml    # 10-step 文件摄入工作流
│       ├── search.workflow.toml # 9-step 智能检索工作流
│       └── ask.workflow.toml    # 6-step RAG 问答工作流
├── plugins/
│   ├── parsers/            # text, code, pdf, data_file, image, url, csv (7)
│   ├── templates/          # paper, code, dataset, note (4)
│   ├── chunkers/           # langchain(默认), paragraph, token, fixed_size, semantic, small_to_big (6)
│   ├── searchers/          # lightrag(默认), keyword, simple (3)
│   ├── indexers/           # sqlite
│   ├── key_managers/       # sqlite (递归CTE, JSON1标签)
│   ├── vector_stores/      # lancedb (IVF_PQ ANN, 模型过滤器)
│   ├── graph_stores/       # sqlite (实体+关系+遍历+孤立清理)
│   ├── entity_extractors/  # llm (pydantic-ai 结构化输出)
│   ├── file_categories/    # docs, data, code, code.py, img, audio, video, web, other (9)
│   ├── importers/exporters/# tar
│   └── syncers/            # rclone
├── cli/                    # Click CLI (20 命令) + formatter.py (ProgressDisplay)
└── utils/                  # config, logging, hash, path, api_errors
```

**注意**: `plugins/llm_providers/` 和 `plugins/embedders/` 已移除 — LLM 通过 `AgentManager`/`AgentRegistry`，Embedding 通过 `core/embedder.py` (`get_embedder()`)。

---

## 开发规则

| # | 规则 | 要点 |
|---|------|------|
| 1 | 先读接口 | 签名/返回值/异常须匹配 ABC |
| 2 | PluginHub | `__plugin_impl__ = True` 自动注册 + `PluginHub.get(IXxx, "name")` 获取; 禁止直接 import 实现 |
| 3 | 优先级 | 内置 < `~/.config/mnemo/plugins/` < `{data_dir}/plugins/` |
| 4 | 错误 | 批量: skip+log; API 429/5xx/timeout: retry 3次+退避, **4xx: fail fast**; 配置: fail early |
| 5 | 路径 | `safe_resolve()` 校验在 `data_dir` 内 |
| 6 | 语言 | 标识符/docstring/注释: EN; CLI help: EN; 提示词: EN (`-zh` 变体可用) |
| 7 | LLM/Embedding | LLM: `AgentManager.get_instance().get_agent("name")`；Embedding: `from mnemo.core.embedder import get_embedder`；调用者使用 pydantic-ai 原生 API |
| 8 | **Workflow Engine** | 所有业务逻辑通过 WorkflowEngine 执行；Step 注册用 `@StepRegistry.register_function`；工作流 TOML 放 `builtin/workflows/`；Agent step 用 `AgentRegistry.get_agent_with_tools()` |

`format_api_error(exc, context)` 映射异常→可操作诊断 (utils/api_errors.py)。`config show/get` 永不解析 `ENV::` 占位符。

---

## 数据目录

```
{data_dir}/
├── raw/{category}/{type}/{chunk}/       原始文件
├── raw_md/{category}/{type}/{chunk}/    Markdown 转换
├── raw_md_parents/                      Small-to-Big 父 chunks (纯文本)
├── raw_wiki/ + raw_metadata/            Wiki + 元信息
├── embedding/             LanceDB: raw_md (chunks), raw_wiki, metadata, raw_md_parents
├── .mnemo/
│   ├── index.db           SQLite 全局索引
│   ├── config.toml + file_categories.toml  项目配置
│   ├── prompts.toml       用户提示词覆盖 (可选)
│   └── trash/             软删除回收站
└── plugins/               项目级插件 (可选)
```

---

## 配置系统

优先级: `MNEMO_*` 环境变量 > `{data_dir}/.mnemo/config.toml` > `~/.config/mnemo/config.toml` > schema 默认值。两份 TOML: `config.toml` (`[interface]`+`[interface.plugin]`) + `file_categories.toml` (`[file_category.name]` 含 `chunker_config` 子节)。`mnemo init` 自动生成模板。

Agent 通过 `[agent.xxx]` 段配置，Embedder 通过单段 `[embedder]` 配置（一个 KB 一个模型）:

```toml
[agent.default]
model = "deepseek-v4-flash"
base_url = "https://api.deepseek.com/v1"
# api_key = ""  # 从 $LLM_API_KEY 读取
temperature = 0.3
max_tokens = 2048

[embedder]
model = "text-embedding-v4"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# api_key = ""  # 从 $DASHSCOPE_API_KEY 读取
batch_size = 10
```

---

## 关键模块状态

| 模块 | 状态 | 实现 |
|------|:----:|------|
| **Workflow Engine** | ✅ | DAG 引擎: graphlib + asyncio + EventBus (85 tests) |
| **AgentRegistry + ToolLibrary** | ✅ | AgentManager 升级 + 4 内置 tools |
| **WorkflowConfigLoader** | ✅ | TOML 加载 + 三层合并 (builtin/global/project) |
| **Add 工作流 (8 steps)** | ✅ | validate → copy → metadata → markdown → wiki → entity → embed → index |
| **Search 工作流 (9 steps)** | ✅ | rewrite → judge → plan → parallel search → fuse → rerank → filter |
| **Ask 工作流 (6 steps)** | ✅ | analyze → search → assemble → generate → verify |
| IChunker (6) | ✅ | langchain (默认), paragraph, token, fixed_size, semantic, small_to_big |
| ISearcher (3) | ✅ | LightRAG (向量+BM25+图谱 RRF), Keyword (BM25+jieba), Simple (grep) |
| AgentManager | ✅ | pydantic-ai Agent 工厂; 委托到 AgentRegistry |
| Embedder | ✅ | `core/embedder.py` 模块级单例; `get_embedder()` 全局获取; `[embedder]` config |
| IVectorStore | ✅ | LanceDB 动态维度 ANN (IVF_PQ), 模型过滤器 |
| IGraphStore | ✅ | SQLite 实体/关系+图遍历+孤立清理 |
| IIndexer / IKeyManager | ✅ | SQLite: CRUD + JSON1 标签 + 递归 CTE 键展开 |
| IEntityExtractor | ✅ | LLM (pydantic-ai 结构化输出) + keyword fallback |
| IParser (7) / ITemplate (4) | ✅ | text/code/pdf/data_file/image/url/csv; paper/code/dataset/note |
| IFileCategory (9) | ✅ | docs/data/code/code.py/img/audio/video/web/other |
| CLI (20 命令) | ✅ | init/add/search/ask/.../serve/mcp; ProgressDisplay 双计时器 |
| MnemoAPI | ✅ | 统一 Python 门面 (api/client.py)；MCP + REST 委托 |
| MCP Server | ✅ | 5 tools: search, ask, list, get, stats |
| REST API | ✅ | FastAPI + OpenAPI (/docs) |
| ParamConfig / PromptManager | ✅ | MRO合并+env_var解析; TOML提示词+用户覆盖 |
| IWatcher/ISyncer/IImporter/IExporter | ✅ | watchdog / rclone / tar |
| AskPipeline | ✅ | RAG: query扩展→search→rerank→context组装→LLM回答+引用 |
| IReorganizer / KB.add_url() | ❌ | stub |

---

## 测试

227 tests collected (221 pass, 6 deselected pre-existing). pytest, 覆盖率目标 >80%.

---

## 常见问题

- **插件放哪？** 个人→`~/.config/mnemo/plugins/`, 项目→`{data_dir}/plugins/`, 贡献→`src/mnemo/plugins/`
- **是否已实现？** `PluginHub.list_names(IXxx)` → 已注册即已实现; `StepRegistry.list_functions()` → 已注册的 workflow steps
- **切换 chunker/searcher？** 在对应 TOML 设 `chunker` / `default_plugin`; searcher 切换后 `add` 自动裁剪管道
- **配置多 agent？** 在 config.toml 添加 `[agent.my_agent]` 段，代码中 `AgentManager.get_instance().get_agent("my_agent")` 或 `AgentRegistry._get().get_agent_with_tools("my_agent", tools=[...])` 获取
- **切换 embedding 模型？** 修改 `[embedder]` 段的 model/base_url，重新 `mnemo add` 摄入文件
- **提示词自定义？** `{data_dir}/.mnemo/prompts.toml` 覆盖，或 `PromptManager.register_prompt()` 运行时注册
- **v1 还是 v2？** 所有功能已迁移到 v2 WorkflowEngine; 旧框架 (pipeline.py, kb_ask.py v1) 已移除
- **自定义工作流？** 创建 `{data_dir}/.mnemo/workflows/<name>.workflow.toml`，自动合并到内置定义中
- **注册新 Step？** `@StepRegistry.register_function("my_step")` 用于 FunctionStep; 或通过 TOML `type = "agent"` 使用 AgentStep
- **调试工作流？** 使用 `CLISink` 查看步骤进度; 使用 `LogSink` 写 JSON 事件日志; 使用 `MetricsSink` 查看 token/latency 统计
- **管道诊断？** `mnemo add --diagnose file.pdf` / `mnemo search --diagnose "query"` → 写入 `<data_dir>/.mnemo/diagnostics/*.jsonl`; 包含每阶段的输入/输出/耗时/向量距离/RRF 评分细节; `--verbose` 同时输出到终端
- **注册新插件？** 接口继承 `PluginBase, ABC` + `__plugin_interface__ = True`; 实现继承接口 + `__plugin_impl__ = True` + `name = "..."`; 无需手动注册
