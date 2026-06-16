"""
Mnemo 核心模块

基于接口 + PluginHub + 依赖注入的插件化架构。
每个组件通过 PluginHub 获取，可被独立实现和替换。
"""

from mnemo.core.kb import KnowledgeBase
from mnemo.core.plugin_base import PluginBase, PluginHub

__all__ = [
    "KnowledgeBase",
    "PluginBase",
    "PluginHub",
]
