"""Multi-format document loading for DocMind AI."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
)

# Map file extensions to loader classes
LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}


def load_file(file_path: str | Path) -> list[Document]:
    """Load a single file into LangChain Documents.

    Args:
        file_path: Path to the file to load.

    Returns:
        List of Document objects.

    Raises:
        ValueError: If the file type is not supported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(LOADER_MAP.keys())}"
        )

    loader = loader_cls(str(path))
    docs = loader.load()

    # Ensure source metadata is set
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = path.name

    return docs


def load_url(url: str) -> list[Document]:
    """Load a web page into LangChain Documents.

    Args:
        url: The URL to load.

    Returns:
        List of Document objects.
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
    return docs


def load_documents(
    file_paths: list[str | Path] | None = None,
    urls: list[str] | None = None,
) -> list[Document]:
    """Load documents from multiple files and/or URLs.

    Args:
        file_paths: List of file paths to load.
        urls: List of URLs to load.

    Returns:
        Combined list of Document objects.
    """
    docs: list[Document] = []

    if file_paths:
        for path in file_paths:
            docs.extend(load_file(path))

    if urls:
        for url in urls:
            docs.extend(load_url(url))

    return docs
