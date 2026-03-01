"""Tests for retriever module (requires embeddings, so these are integration tests)."""

import pytest


def test_retriever_import():
    """Test that the retriever module can be imported."""
    from src.retriever import create_retriever
    assert callable(create_retriever)
