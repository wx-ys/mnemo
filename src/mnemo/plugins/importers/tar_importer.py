"""Tar-based importer for external knowledge bases."""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

from mnemo.core.interfaces import IImporter


class TarImporter(IImporter):
    """Import files from a tar.gz archive into the knowledge base.

    Hash-based deduplication is applied automatically.

    Parameters
    ----------
    kb : KnowledgeBase or None
        The KB instance to import into. Set via ``init()``.
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

    def import_from(self, source: Path, dry_run: bool = False) -> dict:
        """Import files from a tar.gz archive or directory.

        Parameters
        ----------
        source : Path
            Path to a ``.tar.gz`` archive or a directory.
        dry_run : bool, optional
            If True, preview without importing.

        Returns
        -------
        dict
            Report with keys: 'imported', 'skipped', 'errors'.
        """
        if self._kb is None:
            return {"imported": 0, "skipped": 0, "errors": ["No KB bound. Call init(kb) first."]}

        imported = 0
        skipped = 0
        errors: list[str] = []

        if source.suffix == ".gz" or source.name.endswith(".tar.gz"):
            imported, skipped, errors = self._import_tar(source, dry_run)
        elif source.is_dir():
            imported, skipped, errors = self._import_dir(source, dry_run)
        else:
            errors.append(f"Unsupported source format: {source}")

        return {"imported": imported, "skipped": skipped, "errors": errors}

    def _import_tar(
        self, source: Path, dry_run: bool
    ) -> tuple[int, int, list[str]]:
        """Extract and import from tar.gz.

        Parameters
        ----------
        source : Path
            Path to the tar.gz file.
        dry_run : bool

        Returns
        -------
        tuple[int, int, list[str]]
            (imported, skipped, errors).
        """
        imported = 0
        skipped = 0
        errors: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(source, "r:gz") as tar:
                tar.extractall(tmp)

            tmp_path = Path(tmp)
            for f in tmp_path.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    if dry_run:
                        imported += 1
                    else:
                        result = self._kb.add(str(f))
                        if result is not None:
                            imported += 1
                        else:
                            skipped += 1
                except Exception as e:
                    errors.append(f"{f.name}: {e}")

        return imported, skipped, errors

    def _import_dir(
        self, source: Path, dry_run: bool
    ) -> tuple[int, int, list[str]]:
        """Import all files from a directory.

        Parameters
        ----------
        source : Path
            Path to the source directory.
        dry_run : bool

        Returns
        -------
        tuple[int, int, list[str]]
        """
        imported = 0
        skipped = 0
        errors: list[str] = []

        for f in source.rglob("*"):
            if not f.is_file():
                continue
            try:
                if dry_run:
                    imported += 1
                else:
                    result = self._kb.add(str(f))
                    if result is not None:
                        imported += 1
                    else:
                        skipped += 1
            except Exception as e:
                errors.append(f"{f.name}: {e}")

        return imported, skipped, errors
