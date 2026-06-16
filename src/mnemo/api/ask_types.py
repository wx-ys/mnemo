"""Mnemo RAG Ask response types — pydantic models.

Structured types for the ``ask()`` pipeline: query → search → rerank → answer.
"""

from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """A cited source snippet from the knowledge base.

    Parameters
    ----------
    file_id : str
        Source file identifier.
    filename : str
        Source filename.
    snippet : str
        The relevant text passage.
    relevance : float
        Relevance score (0.0–1.0).
    """

    file_id: str = ""
    filename: str = ""
    snippet: str = ""
    relevance: float = 0.0


class AskResponse(BaseModel):
    """RAG question-answering result.

    Parameters
    ----------
    answer : str
        The LLM-generated answer with inline citation markers [1], [2], ...
    citations : list of Citation
        Source citations for each marker.
    grounded : bool
        True if the answer is strictly grounded in the knowledge base.
    model : str
        LLM model used to generate the answer.
    tokens_used : int
        Estimated token consumption.
    """

    answer: str = ""
    citations: list[Citation] = []
    grounded: bool = True
    model: str = ""
    tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
