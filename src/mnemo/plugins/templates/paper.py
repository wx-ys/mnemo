"""Academic paper wiki template — category: 文档."""

from mnemo.plugins.base import BaseTemplate


class PaperTemplate(BaseTemplate):
    """Generate structured summaries for academic papers.

    Produces a wiki with: research question, method, contributions,
    key findings, limitations, keywords, and cross-references.
    """

    __plugin_impl__ = True
    name = "paper"
    category = "docs"
    supported_types = ["pdf", "docx"]

    system_prompt = (
        "你是一个学术论文分析助手。请根据以下论文内容，生成结构化的摘要。"
        "使用中文回答，但保留英文专业术语。"
    )

    user_prompt_template = (
        "## 论文基本信息\n"
        "- 文件名：{filename}\n"
        "- 来源：{source}\n"
        "\n"
        "## 论文内容\n"
        "{content}\n"
        "\n"
        "请生成以下结构的摘要：\n"
        "1. **研究问题**：一句话描述论文要解决的核心问题\n"
        "2. **方法**：核心方法/技术路线/模型架构\n"
        "3. **贡献**：主要贡献点（列表）\n"
        "4. **关键发现**：最重要的实验结果或发现\n"
        "5. **局限性**：论文提到的局限或未来方向\n"
        "6. **关键词**：3-8 个关键词\n"
        "7. **与其他工作的关系**：如果有提到的话\n"
    )
