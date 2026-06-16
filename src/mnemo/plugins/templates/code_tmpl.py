"""Source code wiki template — category: 代码."""

from mnemo.plugins.base import BaseTemplate


class CodeTemplate(BaseTemplate):
    """Generate structured documentation from source code.

    Produces: file purpose, main components, dependencies,
    usage examples, and caveats.
    """

    __plugin_impl__ = True
    name = "code"
    category = "code"
    supported_types = []  # category-level template

    system_prompt = (
        "你是一个代码分析助手。请根据提供的代码文件，"
        "生成结构化的代码说明文档。"
    )

    user_prompt_template = (
        "## 代码文件信息\n"
        "- 文件名：{filename}\n"
        "- 来源：{source}\n"
        "\n"
        "## 代码内容\n"
        "{content}\n"
        "\n"
        "请生成以下结构的代码说明：\n"
        "1. **文件用途**：这个文件/模块的核心功能\n"
        "2. **主要组件**：类、函数、关键变量及其作用\n"
        "3. **依赖关系**：导入的外部库及用途\n"
        "4. **使用示例**：关键函数/类的典型用法\n"
        "5. **注意事项**：边界条件、潜在问题、TODO\n"
    )
