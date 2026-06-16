"""Data file parsers — category: 数据文件.

Covers pure-data formats: CSV, NumPy, HDF5, JSON, Parquet, etc.
Pure data files contain numeric/structured data without complex text.
Default behavior: no markdown conversion (only stats), no wiki, no embedding.
Users can override via CLI flags or config.
"""

from pathlib import Path

from mnemo.plugins.base import BaseParser

# ============================================================================
# Category-level parser (fallback for data file types without a specific parser)
# ============================================================================

class DataFileParser(BaseParser):
    """Default parser for the 数据文件 (data files) category.

    Produces a metadata/statistics summary rather than full content,
    since data files can be very large (multi-GB).
    """

    __plugin_impl__ = True
    name = "data_file"
    category = "data"
    supported_types = []  # category-level only, no specific types

    @property
    def default_enable_md(self) -> bool:
        return False

    @property
    def default_enable_wiki(self) -> bool:
        return False

    @property
    def default_enable_embed(self) -> bool:
        return False

    def parse(self, file_path: Path) -> str:
        """Generate a statistical summary for a data file.

        Parameters
        ----------
        file_path : Path
            Path to the data file.

        Returns
        -------
        str
            Markdown summary.
        """
        stat = file_path.stat()
        ext = file_path.suffix.lower().lstrip(".")

        return "\n".join([
            f"# Data File: {file_path.name}",
            "",
            f"- **Format**: {ext}",
            f"- **Size**: {self._format_size(stat.st_size)}",
            f"- **Modified**: {stat.st_mtime}",
            "",
            "## Note",
            "",
            "This is a pure data file. Full content was not converted.",
            "To add a description, use: `mnemo update --note \"...\"`",
        ])

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format a byte count as a human-readable string.

        Parameters
        ----------
        size_bytes : int
            Size in bytes.

        Returns
        -------
        str
            Formatted string, e.g. '1.5 GB'.
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


# ============================================================================
# Type-level parsers (override category defaults for specific types)
# ============================================================================

class CSVParser(BaseParser):
    """CSV/TSV file parser — type-level override.

    Generates a Markdown table preview plus statistical summary via pandas.
    """

    __plugin_impl__ = True
    name = "csv"
    category = "data"
    supported_types = ["csv", "tsv"]

    @property
    def default_enable_md(self) -> bool:
        return True

    @property
    def default_enable_wiki(self) -> bool:
        return True

    @property
    def default_enable_embed(self) -> bool:
        return True

    def parse(self, file_path: Path) -> str:
        """Convert CSV/TSV to Markdown table + stats.

        Parameters
        ----------
        file_path : Path
            Path to the CSV/TSV file.

        Returns
        -------
        str
            Markdown with table preview and column statistics.
        """
        import pandas as pd

        try:
            sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep, nrows=100)
        except Exception as e:
            return f"# CSV Parse Failed\n\n{file_path.name}: {e}"

        return "\n".join([
            f"# Data Table: {file_path.name}",
            "",
            f"- **Rows**: {len(df)} (showing first 100)",
            f"- **Columns**: {len(df.columns)}",
            f"- **Column names**: {', '.join(df.columns.tolist())}",
            "",
            "## Statistical Summary",
            "",
            df.describe(include="all").to_markdown() if len(df) > 0 else "(empty)",
            "",
            "## Preview (first 20 rows)",
            "",
            df.head(20).to_markdown(),
        ])


class NPYParser(BaseParser):
    """NumPy .npy/.npz file parser — type-level override.

    Generates shape, dtype, and basic statistics for each array.
    """

    __plugin_impl__ = True
    name = "npy"
    category = "data"
    supported_types = ["npy", "npz"]

    @property
    def default_enable_md(self) -> bool:
        return True

    @property
    def default_enable_wiki(self) -> bool:
        return False

    @property
    def default_enable_embed(self) -> bool:
        return False

    def parse(self, file_path: Path) -> str:
        """Generate a statistical summary for a NumPy array.

        Parameters
        ----------
        file_path : Path
            Path to the .npy/.npz file.

        Returns
        -------
        str
            Markdown summary.
        """
        import numpy as np

        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            return f"# NPY Parse Failed\n\n{file_path.name}: {e}"

        lines = [f"# NumPy Array: {file_path.name}", ""]

        if file_path.suffix.lower() == ".npz":
            lines.append(f"- **Arrays**: {len(data.files)}")
            lines.append("")
            for key in data.files:
                arr = data[key]
                lines.append(f"## {key}")
                lines.extend(self._describe_array(arr))
        else:
            lines.extend(self._describe_array(data))

        return "\n".join(lines)

    @staticmethod
    def _describe_array(arr) -> list[str]:
        """Describe a single numpy array.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to describe.

        Returns
        -------
        list of str
            Description lines.
        """
        if not hasattr(arr, 'shape'):
            return [f"- **Type**: {type(arr).__name__}", f"- **Value**: {arr}"]
        return [
            f"- **Shape**: {arr.shape}",
            f"- **dtype**: {arr.dtype}",
            f"- **Size**: {arr.nbytes} bytes",
            f"- **Min**: {arr.min() if arr.size > 0 else 'N/A'}",
            f"- **Max**: {arr.max() if arr.size > 0 else 'N/A'}",
            f"- **Mean**: {arr.mean() if arr.size > 0 else 'N/A':.4f}",
        ]


class HDF5Parser(BaseParser):
    """HDF5 file parser — type-level override.

    Walks the HDF5 group/dataset tree and lists structure.
    """

    __plugin_impl__ = True
    name = "hdf5"
    category = "data"
    supported_types = ["hdf5", "h5", "hdf"]

    @property
    def default_enable_md(self) -> bool:
        return True

    @property
    def default_enable_wiki(self) -> bool:
        return False

    @property
    def default_enable_embed(self) -> bool:
        return False

    def parse(self, file_path: Path) -> str:
        """Generate a structure summary for an HDF5 file.

        Parameters
        ----------
        file_path : Path
            Path to the .h5/.hdf5 file.

        Returns
        -------
        str
            Markdown tree of groups and datasets.
        """
        try:
            import h5py
            f = h5py.File(file_path, "r")
        except Exception as e:
            return f"# HDF5 Parse Failed\n\n{file_path.name}: {e}"

        lines = [f"# HDF5 File: {file_path.name}", ""]

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(
                    f"- **Dataset**: `{name}` — shape={obj.shape}, dtype={obj.dtype}"
                )
            elif isinstance(obj, h5py.Group):
                lines.append(f"- **Group**: `{name}`")

        f.visititems(walk)
        return "\n".join(lines)
