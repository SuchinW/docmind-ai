"""Tests for document loader module."""

import tempfile
from pathlib import Path

import pytest

from src.document_loader import load_file, load_documents


def test_load_text_file():
    """Test loading a plain text file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello, this is a test document.\nIt has two lines.")
        f.flush()
        docs = load_file(f.name)

    assert len(docs) >= 1
    assert "Hello" in docs[0].page_content
    assert "source" in docs[0].metadata


def test_load_unsupported_file():
    """Test that unsupported file types raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file(f.name)


def test_load_documents_empty():
    """Test loading with no inputs returns empty list."""
    docs = load_documents()
    assert docs == []
