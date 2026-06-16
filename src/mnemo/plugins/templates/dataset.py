"""Dataset wiki template — category: 数据文件."""

from mnemo.plugins.base import BaseTemplate


class DatasetTemplate(BaseTemplate):
    """Generate structured data cards for datasets.

    Produces a Data Card with: overview, features, scale, use cases,
    caveats, and relationships to other data.
    """

    __plugin_impl__ = True
    name = "dataset"
    category = "data"
    supported_types = []  # category-level template

    system_prompt = (
        "你是一个数据分析助手。请根据提供的数据文件摘要，"
        "生成结构化的数据卡片 (Data Card)。"
    )

    user_prompt_template = (
        "## 数据文件信息\n"
        "- 文件名：{filename}\n"
        "- 文件类型：{file_type}\n"
        "- 来源：{source}\n"
        "\n"
        "## 文件内容 (摘要/统计信息)\n"
        "{content}\n"
        "\n"
        "请生成以下结构的数据卡片：\n"
        "1. **数据概述**：这个数据集包含什么内容\n"
        "2. **数据特征**：主要变量/列/字段及其含义\n"
        "3. **数据规模**：样本数/特征数/维度\n"
        "4. **数据用途**：这个数据可以用来做什么\n"
        "5. **注意事项**：缺失值/异常值/使用限制\n"
        "6. **与其他数据的关系**：如果有的话\n"
    )
