"""Docs file category — standard document formats."""

from mnemo.core.interfaces.file_category import IFileCategory


class DocsCategory(IFileCategory):
    """Standard office and text documents (PDF, DOCX, TXT, etc.)."""

    __plugin_impl__ = True
    name = "docs"
    parent = None
    types = ["pdf", "docx", "ppt", "pptx", "txt", "md", "rst", "log", "tex"]

    # All config fields inherited from IFileCategory (defaults match)
