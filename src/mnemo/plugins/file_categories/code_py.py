"""Python-specific file category — extends the code category."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class CodePyCategory(IFileCategory):
    """Python source files — specializes the ``code`` category.

    Uses the same parser/template as ``code`` but allows
    Python-specific overrides (e.g., custom wiki prompts).
    """

    __plugin_impl__ = True
    name = "code.py"
    parent = "code"
    types = ["py"]

    config_schema = {
        "parser": Param(type="str", default="code", desc="Parser plugin for Python code"),
        "template": Param(type="str", default="code", desc="Template plugin for Python code wiki"),
        "chunker": Param(type="str", default="token", desc="Chunker plugin (token-based for Python code)"),
    }
