"""File and text hashing utilities."""

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute the hash digest of a file.

    Parameters
    ----------
    file_path : Path
        Path to the file.
    algorithm : str, optional
        Hash algorithm name (sha256, md5, blake2b). Default is 'sha256'.

    Returns
    -------
    str
        Hash string in '{algorithm}:{hex_digest}' format.
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return f"{algorithm}:{hasher.hexdigest()}"


def compute_text_hash(text: str, algorithm: str = "sha256") -> str:
    """Compute the hash digest of a text string.

    Used to detect changes in metadata/markdown content.

    Parameters
    ----------
    text : str
        Text content to hash.
    algorithm : str, optional
        Hash algorithm name. Default is 'sha256'.

    Returns
    -------
    str
        Hash string in '{algorithm}:{hex_digest}' format.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode("utf-8"))
    return f"{algorithm}:{hasher.hexdigest()}"
