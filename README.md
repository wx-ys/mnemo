# Mnemo

> AI-agent-friendly personal knowledge base — External Memory Hub
> 面向 AI Agent 的个人知识库 — 外置记忆中枢

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mnemo is a file-based personal knowledge base. Ingest files → auto-classify → convert to Markdown → LLM wiki summary → embed → hybrid search.

Mnemo 是一个文件化的个人知识库工具。加入文件 → 自动分类 → 转为 Markdown → LLM 生成 Wiki 摘要 → 生成 Embedding → 混合检索。

## Design Philosophy / 设计理念

- **Files ARE the database** — unified directory structure. cp/tar/rsync = migrate/backup/merge.
  **文件即数据库** — 统一目录结构，复制即迁移，tar 即备份。
- **AI-agent-friendly** — CLI JSON output + Python API. Directly callable by OpenClaw, Hermes, etc.
  **AI Agent 友好** — CLI JSON 输出 + Python API，可被 Agent 直接调用。
- **Fully pluggable** — parsers, templates, embedding backends all swappable. Agents implement modules in parallel.
  **全插件化** — 解析器、模板、Embedding 后端全部可插拔，Agent 可并行开发各模块。
- **Category + Type classification** — broad category (文档/数据文件/代码/...) → specific type (pdf/csv/py/...). Parser/template resolved via fallback chain.
  **二级分类** — Category（大类）→ Type（具体类型），Parser/Template 按回退链自动匹配。

## Quick Start / 快速开始

```bash
# Install
pip install mnemo

# Initialize a knowledge base / 初始化知识库
mnemo init ~/my-knowledge

# Add a file / 添加文件
mnemo add --file paper.pdf --keys "research::paper, field::nlp"

# Search / 检索
mnemo search "transformer architecture" --keys "research::paper"

# Python API
from mnemo import KnowledgeBase
kb = KnowledgeBase("~/my-knowledge")
results = kb.search("attention mechanism", keys=["research::paper"])
```

## Project Structure / 项目结构

```
src/mnemo/
├── core/               # Core logic (interfaces, registry, KnowledgeBase)
│   └── interfaces/     # ABC interfaces (split by function)
├── cli/                # CLI (Click)
│   └── commands/       # Subcommand implementations (currently stubs)
├── api/                # Python API (for agent consumption)
├── plugins/            # Built-in plugins
│   ├── parsers/        # File parsers (pdf, code, csv, npy, image, url, text)
│   └── templates/      # Wiki templates (paper, dataset, code, note)
└── utils/              # Utilities (config, logging, hash, path safety)
```

## Plugin Development / 插件开发

```python
from mnemo.core.registry import ParserRegistry
from mnemo.plugins.base import BaseParser

@ParserRegistry.register
class MyParser(BaseParser):
    name = "my_format"
    category = "文档"          # category for directory layout & fallback
    supported_types = ["xyz"]  # file extensions this parser handles

    def parse(self, file_path):
        """Convert .xyz files to Markdown."""
        ...
```

## License / 许可证

MIT
