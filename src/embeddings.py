"""Embedding model management for DocMind AI."""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings(model: str = "text-embedding-3-small") -> Embeddings:
    """Get an embedding model instance.

    Args:
        model: The embedding model name.

    Returns:
        An Embeddings instance.
    """
    return OpenAIEmbeddings(model=model)
