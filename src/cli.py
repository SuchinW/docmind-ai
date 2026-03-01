"""Command-line interface for DocMind AI."""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import create_retriever
from src.chain import get_llm, create_rag_chain, query
from src.memory import ChatHistory


def main():
    parser = argparse.ArgumentParser(
        description="DocMind AI - Document Q&A from the command line"
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        help="Paths to documents or directories to ingest",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="URLs to ingest",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Single question to ask (non-interactive mode)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive chat mode",
    )
    args = parser.parse_args()

    if not args.docs and not args.urls:
        parser.error("Provide at least --docs or --urls")

    config = load_config()

    # Collect file paths
    file_paths = []
    if args.docs:
        for doc_path in args.docs:
            p = Path(doc_path)
            if p.is_dir():
                for ext in [".pdf", ".txt", ".csv", ".md"]:
                    file_paths.extend(p.glob(f"*{ext}"))
            elif p.is_file():
                file_paths.append(p)

    # Load and process documents
    print(f"Loading {len(file_paths)} file(s)...")
    docs = load_documents(file_paths=file_paths, urls=args.urls)
    print(f"Loaded {len(docs)} document(s)")

    chunks = split_documents(
        docs,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    print(f"Split into {len(chunks)} chunks")

    # Build vector store
    print("Building vector store...")
    embeddings = get_embeddings(config["embeddings"]["model"])
    vector_store = create_vector_store(chunks, embeddings)

    # Create retriever and chain
    retriever = create_retriever(
        vector_store,
        chunks,
        top_k=config["retrieval"]["top_k"],
        search_type=config["retrieval"]["search_type"],
    )

    llm = get_llm(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
    )

    chain = create_rag_chain(retriever, llm)
    print("Ready!\n")

    # Single query mode
    if args.query:
        answer = query(chain, args.query)
        print(f"Answer: {answer}")
        return

    # Interactive mode
    history = ChatHistory()
    print("DocMind AI - Interactive Mode (type 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        history.add_user_message(user_input)
        answer = query(chain, user_input, history.get_messages())
        history.add_ai_message(answer)
        print(f"\nDocMind: {answer}\n")


if __name__ == "__main__":
    main()
