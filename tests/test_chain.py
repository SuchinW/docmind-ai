"""Tests for chain module."""

from src.chain import _format_docs
from langchain_core.documents import Document


def test_format_docs():
    """Test document formatting for context."""
    docs = [
        Document(page_content="Content A", metadata={"source": "file1.pdf", "page": 1}),
        Document(page_content="Content B", metadata={"source": "file2.txt"}),
    ]
    result = _format_docs(docs)
    assert "[Source 1: file1.pdf (page 1)]" in result
    assert "[Source 2: file2.txt]" in result
    assert "Content A" in result
    assert "Content B" in result


def test_format_docs_empty():
    """Test formatting empty document list."""
    result = _format_docs([])
    assert result == ""
