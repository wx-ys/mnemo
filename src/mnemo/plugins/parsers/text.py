"""Plain text parser — universal fallback."""

from pathlib import Path

from mnemo.plugins.base import BaseParser


class TextParser(BaseParser):
    """Fallback parser for plain text and unrecognized formats.

    For .md files, returns the content as-is.
    For other text files, wraps content in a code block.
    For binary files, generates a minimal metadata summary.
    """

    __plugin_impl__ = True
    name = "text"
    category = "other"
    supported_types = [
        "txt", "md", "rst", "log", "xml", "yaml", "yml",
        "toml", "cfg", "ini",
    ]

    @property
    def default_enable_wiki(self) -> bool:
        return False

    @property
    def default_enable_embed(self) -> bool:
        return True

    def parse(self, file_path: Path) -> str:
        """Convert a text file to Markdown.

        Parameters
        ----------
        file_path : Path
            Path to the text file.

        Returns
        -------
        str
            Markdown representation.
        """
        # Detect binary files by checking for null bytes in the first 8KB.
        try:
            raw = file_path.read_bytes()
        except Exception:
            return self._binary_fallback(file_path)

        if b"\x00" in raw[:8192]:
            return self._binary_fallback(file_path)

        content = raw.decode("utf-8", errors="replace")

        if file_path.suffix.lower() in (".md", ".markdown"):
            return content

        return f"# Text File: {file_path.name}\n\n```\n{content}\n```"

    @staticmethod
    def _binary_fallback(file_path: Path) -> str:
        """Generate a minimal summary for unreadable binary files.

        Parameters
        ----------
        file_path : Path
            Path to the binary file.

        Returns
        -------
        str
            Markdown metadata summary.
        """
        stat = file_path.stat()
        return (
            f"# Binary File: {file_path.name}\n\n"
            f"- **Type**: unknown binary\n"
            f"- **Size**: {stat.st_size} bytes\n"
            f"- **Extension**: {file_path.suffix}\n"
        )
