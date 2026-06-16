"""Source code file parser — category: 代码."""

from pathlib import Path

from mnemo.plugins.base import BaseParser


class CodeParser(BaseParser):
    """Parse source code files into syntax-highlighted Markdown.

    Wraps source code in fenced code blocks with language tags.
    """

    __plugin_impl__ = True
    name = "code"
    category = "code"
    supported_types = [
        "py", "js", "ts", "rs", "go", "java", "cpp", "c",
        "h", "sh", "sql", "r", "jl", "swift", "kt",
    ]

    def parse(self, file_path: Path) -> str:
        """Convert a code file to Markdown.

        Parameters
        ----------
        file_path : Path
            Path to the source file.

        Returns
        -------
        str
            Markdown with fenced code block, line count, and size.
        """
        code = file_path.read_text(encoding="utf-8", errors="replace")
        ext = file_path.suffix.lstrip(".")

        return "\n".join([
            f"# Code File: {file_path.name}",
            "",
            f"- **Language**: {ext}",
            f"- **Lines**: {len(code.splitlines())}",
            f"- **Size**: {file_path.stat().st_size} bytes",
            "",
            "## Source Code",
            "",
            f"```{ext}",
            code,
            "```",
        ])
