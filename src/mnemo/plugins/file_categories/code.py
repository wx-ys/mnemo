"""Code file category — source code files."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class CodeCategory(IFileCategory):
    """Source code files in various programming languages."""

    __plugin_impl__ = True
    name = "code"
    parent = None
    types = ["py", "js", "ts", "rs", "go", "java", "c", "cpp", "h",
             "sh", "bash", "sql", "r", "jl"]

    config_schema = {
        "parser": Param(
            type="str", default="code", desc="Parser plugin for code markdown conversion"),
        "template": Param(
            type="str", default="code", desc="Template plugin for code wiki generation"),
        "chunker": Param(
            type="str", default="token", desc="Chunker plugin (token-based for code)"),
    }
