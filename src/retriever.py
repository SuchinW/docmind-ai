"""Hybrid retrieval (BM25 + vector search) for DocMind AI."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# All supported retrieval methods
RETRIEVAL_METHODS = [
    "similarity",
    "mmr",
    "hybrid",
    "rerank",
    "multi_query",
    "contextual_compression",
    "parent_document",
]


def create_retriever(
    vector_store: FAISS,
    chunks: list[Document],
    top_k: int = 4,
    search_type: str = "hybrid",
    llm: BaseChatModel | None = None,
    original_docs: list[Document] | None = None,
) -> BaseRetriever:
    """Create a retriever from a vector store.

    Args:
        vector_store: The FAISS vector store.
        chunks: Original document chunks (needed for BM25 in hybrid mode).
        top_k: Number of documents to retrieve.
        search_type: One of the RETRIEVAL_METHODS.
        llm: Chat model (required for multi_query, contextual_compression).
        original_docs: Full documents before chunking (for parent_document).

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

    if search_type == "rerank":
        return _create_rerank_retriever(vector_store, top_k)

    if search_type == "multi_query":
        if llm is None:
            raise ValueError("multi_query retrieval requires an llm argument")
        return _create_multi_query_retriever(vector_store, llm, top_k)

    if search_type == "contextual_compression":
        if llm is None:
            raise ValueError(
                "contextual_compression retrieval requires an llm argument"
            )
        return _create_contextual_compression_retriever(
            vector_store, llm, top_k
        )

    if search_type == "parent_document":
        return _create_parent_document_retriever(
            vector_store, chunks, original_docs, top_k
        )

    # Default: similarity search
    return vector_store.as_retriever(search_kwargs={"k": top_k})


# ---------------------------------------------------------------------------
# Re-rank retriever: fetch more candidates, re-score with a cross-encoder
# ---------------------------------------------------------------------------

def _create_rerank_retriever(
    vector_store: FAISS,
    top_k: int,
) -> BaseRetriever:
    """Retrieve extra candidates and re-rank with a cross-encoder."""
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k * 3}  # over-fetch candidates
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = CrossEncoderReranker(
        model=cross_encoder, top_n=top_k
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )


# ---------------------------------------------------------------------------
# Multi-query: LLM generates query variations for broader recall
# ---------------------------------------------------------------------------

def _create_multi_query_retriever(
    vector_store: FAISS,
    llm: BaseChatModel,
    top_k: int,
) -> BaseRetriever:
    """LLM generates alternative queries, results are combined."""
    from langchain.retrievers.multi_query import MultiQueryRetriever

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )


# ---------------------------------------------------------------------------
# Contextual compression: LLM extracts only relevant parts of chunks
# ---------------------------------------------------------------------------

def _create_contextual_compression_retriever(
    vector_store: FAISS,
    llm: BaseChatModel,
    top_k: int,
) -> BaseRetriever:
    """LLM extracts only the relevant portions from each retrieved chunk."""
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )

    compressor = LLMChainExtractor.from_llm(llm)

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


# ---------------------------------------------------------------------------
# Parent document: small chunks for retrieval, larger chunks for context
# ---------------------------------------------------------------------------

class _ParentDocumentRetriever(BaseRetriever):
    """Retrieve using small chunks but return the larger parent chunk."""

    child_retriever: BaseRetriever
    parent_lookup: dict[str, Document]  # child_id -> parent doc
    top_k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        child_docs = self.child_retriever.invoke(query)
        seen_parent_ids: set[str] = set()
        parent_docs: list[Document] = []
        for child in child_docs:
            pid = child.metadata.get("parent_id", "")
            if pid and pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                if pid in self.parent_lookup:
                    parent_docs.append(self.parent_lookup[pid])
            elif not pid:
                parent_docs.append(child)
            if len(parent_docs) >= self.top_k:
                break
        return parent_docs


def _create_parent_document_retriever(
    vector_store: FAISS,
    chunks: list[Document],
    original_docs: list[Document] | None,
    top_k: int,
) -> BaseRetriever:
    """Small chunks for retrieval, larger parent chunks for context.

    If original_docs is provided, each small chunk maps back to its parent
    document. Otherwise, we group consecutive chunks as an approximation.
    """
    import hashlib
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Build parent chunks (larger) from original docs or existing chunks
    parents: list[Document] = []
    if original_docs:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        parents = parent_splitter.split_documents(original_docs)
    else:
        # Fallback: treat existing chunks as both parent and child
        parents = chunks

    # Give each parent a stable id
    parent_lookup: dict[str, Document] = {}
    for i, parent in enumerate(parents):
        pid = hashlib.md5(
            f"{parent.metadata.get('source', '')}_{i}".encode()
        ).hexdigest()
        parent.metadata["parent_id"] = pid
        parent_lookup[pid] = parent

    # Create small child chunks and tag them with parent_id
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    child_chunks: list[Document] = []
    for parent in parents:
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent.metadata["parent_id"]
            child_chunks.append(child)

    # Build a temporary FAISS index from child chunks using the same
    # embeddings as the main vector store
    child_vs = FAISS.from_documents(
        child_chunks, vector_store.embedding_function
    )
    child_retriever = child_vs.as_retriever(
        search_kwargs={"k": top_k * 2}  # over-fetch for dedup
    )

    return _ParentDocumentRetriever(
        child_retriever=child_retriever,
        parent_lookup=parent_lookup,
        top_k=top_k,
    )
