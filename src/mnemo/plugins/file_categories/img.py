"""Image file category — raster and vector image formats."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class ImgCategory(IFileCategory):
    """Image files (JPEG, PNG, SVG, etc.)."""

    __plugin_impl__ = True
    name = "img"
    parent = None
    types = ["jpg", "jpeg", "png", "gif", "svg", "webp", "bmp", "tiff"]

    config_schema = {
        "parser": Param(type="str", default="image", desc="Parser plugin for image description"),
        "auto_wiki": Param(type="bool", default=False, desc="Auto-generate wiki on add"),
    }
