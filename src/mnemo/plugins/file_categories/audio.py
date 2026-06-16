"""Audio file category — audio formats."""

from mnemo.core.interfaces.file_category import IFileCategory
from mnemo.core.interfaces.param_spec import Param


class AudioCategory(IFileCategory):
    """Audio files (MP3, WAV, FLAC, etc.)."""

    __plugin_impl__ = True
    name = "audio"
    parent = None
    types = ["mp3", "wav", "flac", "ogg", "m4a", "aac"]

    config_schema = {
        "parser": Param(
            type="str", default="audio", desc="Parser plugin for audio metadata extraction"),
        "auto_wiki": Param(
            type="bool", default=False, desc="Auto-generate wiki on add"),
    }
