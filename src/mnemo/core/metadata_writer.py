"""Metadata file writer.

Generates ``raw_metadata/{category}/{type}/{chunk}/{filename}.md`` files
with YAML front matter and a Markdown body for user notes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class MetadataWriter:
    """Writes structured metadata files for ingested files.

    Each file is a ``.md`` document with:
    - YAML front matter (between ``---`` fences) containing all structured fields.
    - Markdown body for user notes and free-form content.

    Parameters
    ----------
    data_dir : Path
        Root data directory. Metadata is written under
        ``data_dir/raw_metadata/``.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        file_id: str,
        meta: dict[str, Any],
        *,
        md_info: dict[str, Any] | None = None,
        wiki_info: dict[str, Any] | None = None,
    ) -> Path:
        """Write a metadata file.

        Parameters
        ----------
        file_id : str
            Unique file identifier.
        meta : dict
            Core metadata (file_type, filename, file_hash, file_size,
            source_path, tags, keywords, category, keys, etc.).
        md_info : dict, optional
            Markdown conversion info (parser name, char count, line count, etc.).
        wiki_info : dict, optional
            Wiki generation info (template name, model used, token count, etc.).

        Returns
        -------
        Path
            Absolute path to the written metadata file.
        """
        now = datetime.now(UTC).isoformat()

        front_matter = {
            "id": file_id,
            "file_type": meta.get("file_type", ""),
            "filename": meta.get("filename", ""),
            "file_hash": meta.get("file_hash", ""),
            "file_size": meta.get("file_size", 0),
            "source_path": meta.get("source_path", ""),
            "category": meta.get("category", ""),
            "tags": meta.get("tags", []),
            "keys": meta.get("keys", []),
            "keywords": meta.get("keywords", []),
            "added_at": meta.get("added_at", now),
            "updated_at": meta.get("updated_at", now),
            # Processing config
            "config": {
                "auto_md": meta.get("auto_md", True),
                "auto_wiki": meta.get("auto_wiki", True),
                "auto_embed": meta.get("auto_embed", True),
                "parser": meta.get("parser_name", ""),
                "template": meta.get("template_name", ""),
            },
            # Markdown conversion status
            "md": {
                "status": md_info.get("status", "skipped") if md_info else "skipped",
                "generated_at": md_info.get("generated_at", "") if md_info else "",
                "parser": md_info.get("parser", "") if md_info else "",
                "chars": md_info.get("chars", 0) if md_info else 0,
                "lines": md_info.get("lines", 0) if md_info else 0,
                "file_size": md_info.get("file_size", 0) if md_info else 0,
            },
            # Wiki generation status
            "wiki": {
                "status": wiki_info.get("status", "skipped") if wiki_info else "skipped",
                "generated_at": wiki_info.get("generated_at", "") if wiki_info else "",
                "template": wiki_info.get("template", "") if wiki_info else "",
                "model": wiki_info.get("model", "") if wiki_info else "",
                "chars": wiki_info.get("chars", 0) if wiki_info else 0,
                "tokens_used": wiki_info.get("tokens_used", 0) if wiki_info else 0,
                "tokens_input": wiki_info.get("tokens_input", 0) if wiki_info else 0,
                "tokens_output": wiki_info.get("tokens_output", 0) if wiki_info else 0,
            },
            # Embedding status
            "embedding": {
                "raw": meta.get("embed_raw", "skipped"),
                "md": meta.get("embed_md", "skipped"),
                "wiki": meta.get("embed_wiki", "skipped"),
                "metadata": meta.get("embed_metadata", "skipped"),
            },
            # Change log
            "change_log": [
                {
                    "time": now,
                    "action": "add",
                    "detail": "Initial ingestion",
                }
            ],
            # Custom fields (user-extensible)
            "custom": meta.get("custom", {}),
            "source_kb": meta.get("source_kb", ""),
            "version": meta.get("version", 1),
            "related_files": meta.get("related_files", []),
        }

        # Determine output path
        category = meta.get("category", "other")
        file_type = meta.get("file_type", "unknown")
        chunk = meta.get("chunk", "default")
        filename = meta.get("filename", "unknown")

        out_dir = self._data_dir / "raw_metadata" / category / file_type / chunk
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(filename).stem}.md"

        # Write as YAML front matter + markdown body
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            yaml.dump(
                front_matter,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
            f.write("---\n\n")
            f.write("# User Notes\n\n")
            note = meta.get("note", "")
            if note:
                f.write(note + "\n")

        return out_path

    def read(self, file_path: Path) -> dict[str, Any]:
        """Read a metadata file and return the parsed front matter.

        Parameters
        ----------
        file_path : Path
            Path to the ``.md`` metadata file.

        Returns
        -------
        dict
            Parsed YAML front matter.
        """
        text = file_path.read_text(encoding="utf-8")
        parts = text.split("---\n")
        if len(parts) >= 2:
            return yaml.safe_load(parts[1]) or {}
        return {}

    def update_note(self, file_id: str, old_note: str, new_note: str) -> None:
        """Update the user note for a metadata file.

        Reads the existing metadata file, appends to the change log,
        and rewrites with the new note.

        Parameters
        ----------
        file_id : str
            File identifier.
        old_note : str
            Previous note text.
        new_note : str
            New note text (replaces old note).
        """
        # Find the metadata file by scanning raw_metadata/ for the
        # file whose front-matter ``id`` field matches *file_id*.
        meta_dir = self._data_dir / "raw_metadata"
        for md_file in meta_dir.rglob("*.md"):
            front_matter = self.read(md_file)
            if front_matter.get("id") != file_id:
                continue
            # Found it

            # Append change log entry
            now = datetime.now(UTC).isoformat()
            change_log = front_matter.get("change_log", [])
            change_log.append({
                "time": now,
                "action": "update_note",
                "detail": "User note updated",
            })
            front_matter["change_log"] = change_log

            # Rebuild the file
            with open(md_file, "w", encoding="utf-8") as f:
                f.write("---\n")
                yaml.dump(
                    front_matter,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
                f.write("---\n\n")
                f.write("# User Notes\n\n")
                f.write(new_note + "\n")

            return
        # File not found — create a minimal metadata note entry
        # (fallback: note is stored in the index custom field, so this is best-effort)
