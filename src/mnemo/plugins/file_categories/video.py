"""Video file category — video formats."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class VideoCategory(IFileCategory):
    """Video files (MP4, MKV, AVI, etc.)."""

    __plugin_impl__ = True
    name = "video"
    parent = None
    types = ["mp4", "mkv", "avi", "mov", "webm"]

    config_schema = {
        "parser": Param(
            type="str", default="video",
            desc="Parser plugin for video metadata extraction"
        ),
        "auto_md": Param(
            type="bool", default=False,
            desc="Auto-generate markdown on add"
        ),
        "auto_wiki": Param(
            type="bool", default=False,
            desc="Auto-generate wiki on add"
        ),
        "auto_embed": Param(
            type="bool", default=False,
            desc="Auto-generate embedding on add"
        ),
    }
