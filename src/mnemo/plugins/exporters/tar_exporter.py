"""Tar-based exporter for the knowledge base."""

from __future__ import annotations

import tarfile
from pathlib import Path

from mnemo.core.interfaces import IExporter


class TarExporter(IExporter):
    """Export the knowledge base as a self-contained tar.gz archive.

    Parameters
    ----------
    kb : KnowledgeBase or None
        The KB instance to export from. Set via ``init()``.
    """

    __plugin_impl__ = True
    name = "default"

    def __init__(self):
        self._kb = None

    def init(self, kb) -> None:
        """Bind to a KnowledgeBase instance.

        Parameters
        ----------
        kb : KnowledgeBase
        """
        self._kb = kb

    def export_to(
        self,
        dest: Path,
        file_type: str | None = None,
        keys: list[str] | None = None,
        after: str | None = None,
    ) -> Path:
        """Export the knowledge base to a tar.gz archive.

        Parameters
        ----------
        dest : Path
            Destination file path (``.tar.gz`` suffix recommended).
        file_type : str, optional
            Export only this file type.
        keys : list of str, optional
            Export only files matching these keys.
        after : str, optional
            ISO 8601 date — only files added after this date.

        Returns
        -------
        Path
            Path to the created archive.
        """
        if self._kb is None:
            raise RuntimeError("No KB bound. Call init(kb) first.")

        dest = dest.with_suffix(".tar.gz") if dest.suffix != ".gz" else dest

        # Get list of files to export
        files = self._kb.list_files(
            file_type=file_type,
            keys=keys,
            date_from=after,
            limit=100000,
        )

        with tarfile.open(dest, "w:gz") as tar:
            for f in files:
                # Add raw file
                if f.raw_path:
                    raw_file = self._kb.data_dir / f.raw_path
                    if raw_file.exists():
                        tar.add(str(raw_file), arcname=f"raw/{f.filename}")

                # Add markdown file
                if f.md_path:
                    md_file = self._kb.data_dir / f.md_path
                    if md_file.exists():
                        tar.add(str(md_file), arcname=f"raw_md/{f.filename}.md")

                # Add metadata file
                if f.metadata_path:
                    meta_file = self._kb.data_dir / f.metadata_path
                    if meta_file.exists():
                        tar.add(str(meta_file), arcname=f"raw_metadata/{f.filename}.meta.md")

                # Add wiki file
                if f.wiki_path:
                    wiki_file = self._kb.data_dir / f.wiki_path
                    if wiki_file.exists():
                        tar.add(str(wiki_file), arcname=f"raw_wiki/{f.filename}.wiki.md")

        return dest
