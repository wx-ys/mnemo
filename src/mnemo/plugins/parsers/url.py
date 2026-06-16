"""Web page / URL bookmark parser."""

from pathlib import Path

from mnemo.plugins.base import BaseParser


class URLParser(BaseParser):
    """Parse HTML files and URL bookmarks to Markdown.

    For ``.html`` files, extracts body text via BeautifulSoup.
    For ``.url`` bookmark files, reads the target URL.
    """

    __plugin_impl__ = True
    name = "url"
    category = "web"
    supported_types = ["html", "htm", "url"]

    def parse(self, file_path: Path) -> str:
        """Parse an HTML file or URL bookmark.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        str
            Markdown representation.
        """
        if file_path.suffix.lower() == ".url":
            return self._parse_url_bookmark(file_path)
        return self._parse_html(file_path)

    @staticmethod
    def _parse_html(file_path: Path) -> str:
        """Extract text content from an HTML file.

        Parameters
        ----------
        file_path : Path
            Path to the .html file.

        Returns
        -------
        str
            Extracted text as Markdown.
        """
        try:
            from bs4 import BeautifulSoup
            html = file_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            title = soup.title.string if soup.title else file_path.name
            body = soup.get_text("\n", strip=True)

            if len(body) > 50000:
                body = body[:50000] + "\n\n...(truncated)"

            return f"# {title}\n\n{body}"
        except Exception as e:
            return f"# HTML Parse Failed\n\n{file_path.name}: {e}"

    @staticmethod
    def _parse_url_bookmark(file_path: Path) -> str:
        """Read a .url bookmark file.

        Parameters
        ----------
        file_path : Path
            Path to the .url file.

        Returns
        -------
        str
            Markdown with the target URL.
        """
        url = ""
        for line in file_path.read_text(errors="replace").splitlines():
            if line.startswith("URL="):
                url = line[4:]
                break
        return f"# URL Bookmark\n\n- **File**: {file_path.name}\n- **URL**: {url}"
