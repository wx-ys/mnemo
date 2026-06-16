"""Other file category — fallback for unrecognized file types."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class OtherCategory(IFileCategory):
    """Fallback category for unrecognized file types.

    This is the last-resort category used when no other category
    matches a file extension. Processing is conservative by default.
    """

    __plugin_impl__ = True
    name = "other"
    parent = None
    types = []

    config_schema = {
        "auto_md": Param(type="bool", default=False, desc="Auto-generate markdown on add"),
        "auto_wiki": Param(type="bool", default=False, desc="Auto-generate wiki on add"),
        "auto_embed": Param(type="bool", default=False, desc="Auto-generate embedding on add"),
    }
