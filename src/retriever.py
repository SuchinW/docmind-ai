"""Hybrid retrieval (BM25 + vector search) for DocMind AI."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


def create_retriever(
    vector_store: FAISS,
    chunks: list[Document],
    top_k: int = 4,
    search_type: str = "hybrid",
) -> BaseRetriever:
    """Create a retriever from a vector store.

    Args:
        vector_store: The FAISS vector store.
        chunks: Original document chunks (needed for BM25 in hybrid mode).
        top_k: Number of documents to retrieve.
        search_type: One of "similarity", "mmr", or "hybrid".

    Returns:
        A retriever instance.
    """
    if search_type == "hybrid" and chunks:
        # Hybrid: BM25 (keyword) + FAISS (semantic) with RRF
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = top_k

        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6],  # Weight semantic search higher
        )

    if search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "lambda_mult": 0.7},
        )

    # Default: similarity search
    return vector_store.as_retriever(search_kwargs={"k": top_k})
