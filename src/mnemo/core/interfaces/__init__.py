"""Mnemo core interfaces.

Each interface lives in its own module. This package re-exports
all symbols so that ``from mnemo.core.interfaces import IParser``
continues to work.
"""

from mnemo.core.interfaces.chunker import IChunker
from mnemo.core.interfaces.config_loader import IConfigLoader
from mnemo.core.interfaces.entity_extractor import IEntityExtractor
from mnemo.core.interfaces.exporter import IExporter
from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.graph_store import IGraphStore
from mnemo.core.interfaces.importer import IImporter
from mnemo.core.interfaces.indexer import IIndexer
from mnemo.core.interfaces.key_manager import IKeyManager
from mnemo.core.interfaces.param_spec import Param
from mnemo.core.interfaces.parser import IParser
from mnemo.core.interfaces.reorganizer import IReorganizer
from mnemo.core.interfaces.searcher import ISearcher
from mnemo.core.interfaces.syncer import ISyncer
from mnemo.core.interfaces.template import ITemplate, WikiResultProtocol
from mnemo.core.interfaces.types import ChunkInfo, FileMeta, GroupedResult, SearchResult
from mnemo.core.interfaces.vector_store import IVectorStore
from mnemo.core.interfaces.watcher import IWatcher

__all__ = [
    # Types
    "FileMeta",
    "SearchResult",
    "GroupedResult",
    "ChunkInfo",
    "Param",
    # Interfaces
    "IParser",
    "ITemplate",
    "WikiResultProtocol",
    "IIndexer",
    "IKeyManager",
    "ISearcher",
    "IWatcher",
    "ISyncer",
    "IImporter",
    "IExporter",
    "IReorganizer",
    "IConfigLoader",
    "IVectorStore",
    "IGraphStore",
    "IEntityExtractor",
    "IFileCategory",
    "IChunker",
]
