"""
Mnemo Python API — 面向 Agent 的知识库接口

这是 Agent 调用 Mnemo 的推荐入口。
所有方法映射到 KnowledgeBase 核心类。
"""

from __future__ import annotations

from mnemo.api.types import (
    CheckReport,
    FileContext,
    FileInfo,
    ImportReport,
    KnowledgeBaseStats,
    SearchMode,
    SearchResult,
    SyncReport,
)
from mnemo.core.kb import KnowledgeBase

# Re-export for convenience
__all__ = [
    "KnowledgeBase",
    "SearchMode",
    "SearchResult",
    "FileInfo",
    "FileContext",
    "KnowledgeBaseStats",
    "CheckReport",
    "ImportReport",
    "SyncReport",
]
