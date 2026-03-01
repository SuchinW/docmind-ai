"""FAISS vector store operations for DocMind AI."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(
    docs: list[Document],
    embeddings: Embeddings,
) -> FAISS:
    """Create a FAISS vector store from documents.

    Args:
        docs: Chunked documents to embed and store.
        embeddings: Embedding model to use.

    Returns:
        A FAISS vector store instance.
    """
    return FAISS.from_documents(docs, embeddings)


def save_vector_store(vector_store: FAISS, path: str | Path) -> None:
    """Save a vector store to disk.

    Args:
        vector_store: The FAISS vector store to save.
        path: Directory path to save to.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(path: str | Path, embeddings: Embeddings) -> FAISS:
    """Load a vector store from disk.

    Args:
        path: Directory path to load from.
        embeddings: Embedding model (must match the one used to create the store).

    Returns:
        A FAISS vector store instance.
    """
    return FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )
