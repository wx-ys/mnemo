"""Web file category — HTML and web-related formats."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class WebCategory(IFileCategory):
    """Web content files (HTML, URL shortcuts, etc.)."""

    __plugin_impl__ = True
    name = "web"
    parent = None
    types = ["html", "htm", "url"]

    config_schema = {
        "parser": Param(type="str", default="url", desc="Parser plugin for HTML/URL content"),
    }
