"""Data file category — structured data formats (CSV, JSON, HDF5, etc.)."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class DataCategory(IFileCategory):
    """Structured and semi-structured data files."""

    __plugin_impl__ = True
    name = "data"
    parent = None
    types = ["csv", "tsv", "npy", "npz", "hdf5", "h5", "json", "xml",
             "yaml", "yml", "parquet", "pickle", "mat", "npz"]

    config_schema = {
        "template": Param(type="str", default="dataset", desc="Template plugin for dataset wiki generation"),
        "auto_md": Param(type="bool", default=False, desc="Auto-generate markdown on add"),
        "auto_wiki": Param(type="bool", default=False, desc="Auto-generate wiki on add"),
        "auto_embed": Param(type="bool", default=False, desc="Auto-generate embedding on add"),
    }
