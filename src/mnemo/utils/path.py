"""Path safety utilities.

Prevents path traversal attacks and ensures all file operations
stay within the designated data directory.
"""

from __future__ import annotations

from pathlib import Path


class PathSecurityError(Exception):
    """Raised when a path escapes the allowed base directory."""
    pass


def safe_resolve(path: str | Path, base_dir: Path) -> Path:
    """Resolve a path safely, ensuring it does not escape *base_dir*.

    Parameters
    ----------
    path : str or Path
        Relative path to resolve.
    base_dir : Path
        The allowed root directory.

    Returns
    -------
    Path
        Resolved absolute path within *base_dir*.

    Raises
    ------
    PathSecurityError
        If the resolved path lies outside *base_dir*.
    """
    base = base_dir.resolve()
    resolved = (base / path).resolve()

    if not str(resolved).startswith(str(base)):
        raise PathSecurityError(
            f"Path traversal detected: '{path}' resolves to "
            f"'{resolved}', which is outside '{base}'."
        )
    return resolved


def is_within(data_dir: Path, path: Path) -> bool:
    """Check whether *path* is inside *data_dir*.

    Parameters
    ----------
    data_dir : Path
        Root directory.
    path : Path
        Path to check.

    Returns
    -------
    bool
    """
    try:
        path.resolve().relative_to(data_dir.resolve())
        return True
    except ValueError:
        return False


def find_data_dir(start: Path | None = None) -> Path | None:
    """Walk upward from *start* to find a ``.mnemo/`` directory.

    Analogous to how git finds ``.git/``.

    Parameters
    ----------
    start : Path, optional
        Directory to start searching from. Defaults to cwd.

    Returns
    -------
    Path or None
        The data directory, or None if not found.
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / ".mnemo").is_dir():
            return parent
    return None
