"""Generic note wiki template — universal fallback."""

from mnemo.plugins.base import BaseTemplate


class NoteTemplate(BaseTemplate):
    """Fallback template for all file types.

    Produces a simple summary: core content, key points,
    keywords, and additional notes. Auto-detects response language.
    """

    __plugin_impl__ = True
    name = "note"
    category = ""            # fallback — not tied to any category
    supported_types = []     # not tied to any specific type

    system_prompt = (
        "你是一个信息整理助手。请根据提供的内容，生成简洁的摘要。"
        "根据内容语言自动选择回答语言。"
    )

    user_prompt_template = (
        "## 文件信息\n"
        "- 文件名：{filename}\n"
        "- 文件类型：{file_type}\n"
        "\n"
        "## 内容\n"
        "{content}\n"
        "\n"
        "请生成以下结构的摘要：\n"
        "1. **核心内容**：一句话概述\n"
        "2. **关键要点**：3-5 个关键信息点\n"
        "3. **关键词**：3-8 个关键词\n"
        "4. **备注**：任何值得注意的额外信息\n"
    )
