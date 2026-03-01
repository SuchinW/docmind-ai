"""Tests for text splitter module."""

from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_split_short_document():
    """Test that short documents are not split."""
    docs = [Document(page_content="Short text.", metadata={"source": "test"})]
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_split_long_document():
    """Test that long documents are split into multiple chunks."""
    long_text = "This is a sentence. " * 200  # ~4000 chars
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    # All chunks should have metadata preserved
    for chunk in chunks:
        assert chunk.metadata["source"] == "test"


def test_chunk_overlap():
    """Test that chunks have overlapping content."""
    text = "\n\n".join([f"Paragraph {i}. " * 20 for i in range(10)])
    docs = [Document(page_content=text, metadata={"source": "test"})]
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 2
