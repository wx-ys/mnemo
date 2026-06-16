"""Image file parser."""

from pathlib import Path

from mnemo.plugins.base import BaseParser


class ImageParser(BaseParser):
    """Parse image files to metadata descriptions.

    Extracts dimensions, format, and EXIF data via Pillow.
    Visual content description requires an optional vision model.
    """

    __plugin_impl__ = True
    name = "image"
    category = "img"
    supported_types = ["jpg", "jpeg", "png", "gif", "svg", "webp", "bmp", "tiff"]

    @property
    def default_enable_wiki(self) -> bool:
        return False  # vision model needed, disabled by default

    def parse(self, file_path: Path) -> str:
        """Extract image metadata.

        Parameters
        ----------
        file_path : Path
            Path to the image file.

        Returns
        -------
        str
            Markdown with dimensions, format, mode, and EXIF data.
        """
        from PIL import Image

        try:
            img = Image.open(file_path)
            info = [
                f"# Image File: {file_path.name}",
                "",
                f"- **Format**: {img.format}",
                f"- **Dimensions**: {img.size[0]} x {img.size[1]} px",
                f"- **Mode**: {img.mode}",
                f"- **Size**: {file_path.stat().st_size} bytes",
            ]

            exif = img.getexif()
            if exif:
                info.append("")
                info.append("## EXIF Data")
                info.append("")
                for k, v in exif.items():
                    if k != 37500:  # skip MakerNote
                        info.append(f"- {k}: {v}")

            return "\n".join(info)
        except Exception as e:
            return f"# Image Parse Failed\n\n{file_path.name}: {e}"
