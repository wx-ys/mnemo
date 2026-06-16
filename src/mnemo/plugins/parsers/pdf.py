"""PDF document parser."""

from pathlib import Path

from mnemo.plugins.base import BaseParser


class PDFParser(BaseParser):
    """Parse PDF files to Markdown text.

    Uses the ``markitdown`` library for conversion.
    """

    __plugin_impl__ = True
    name = "pdf"
    category = "docs"
    supported_types = ["pdf", "PDF"]

    def parse(self, file_path: Path) -> str:
        """Convert a PDF file to Markdown.

        Parameters
        ----------
        file_path : Path
            Path to the PDF file.

        Returns
        -------
        str
            Markdown representation.
        """
        # TODO: implement with markitdown or pdfplumber
        raise NotImplementedError("PDF parser is not yet implemented")
