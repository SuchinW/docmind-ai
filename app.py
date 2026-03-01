"""DocMind AI - Streamlit Web Application."""

import tempfile
from pathlib import Path

import streamlit as st

from src.config import load_config
from src.document_loader import load_documents, LOADER_MAP
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store
from src.retriever import create_retriever
from src.chain import get_llm, create_rag_chain, query
from src.memory import ChatHistory

# --- Page Config ---
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
)

# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed" not in st.session_state:
    st.session_state.processed = False


def process_documents(files, urls, config):
    """Process uploaded documents into a RAG chain."""
    file_paths = []

    # Save uploaded files to temp directory
    if files:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in files:
            temp_path = Path(temp_dir) / uploaded_file.name
            temp_path.write_bytes(uploaded_file.read())
            file_paths.append(temp_path)

    url_list = [u.strip() for u in urls.split("\n") if u.strip()] if urls else None

    if not file_paths and not url_list:
        st.error("Please upload files or enter URLs.")
        return

    with st.spinner("Loading documents..."):
        docs = load_documents(file_paths=file_paths, urls=url_list)
        st.info(f"Loaded {len(docs)} document(s)")

    with st.spinner("Splitting into chunks..."):
        chunks = split_documents(
            docs,
            chunk_size=config["chunking"]["chunk_size"],
            chunk_overlap=config["chunking"]["chunk_overlap"],
        )
        st.info(f"Created {len(chunks)} chunks")

    with st.spinner("Building vector store & retriever..."):
        embeddings = get_embeddings(config["embeddings"]["model"])
        vector_store = create_vector_store(chunks, embeddings)

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

        st.session_state.chain = create_rag_chain(retriever, llm)
        st.session_state.processed = True
        st.session_state.messages = []
        st.session_state.chat_history = ChatHistory()

    st.success("Documents processed! Ask questions below.")


# --- Sidebar ---
with st.sidebar:
    st.title("DocMind AI")
    st.markdown("*Intelligent Document Q&A*")
    st.divider()

    config = load_config()

    # LLM Settings
    st.subheader("LLM Settings")
    provider = st.selectbox(
        "Provider",
        ["openai", "anthropic"],
        index=0 if config["llm"]["provider"] == "openai" else 1,
    )
    config["llm"]["provider"] = provider

    default_models = {
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001", "claude-opus-4-20250514"],
    }
    model = st.selectbox("Model", default_models.get(provider, ["gpt-4o-mini"]))
    config["llm"]["model"] = model

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    config["llm"]["temperature"] = temperature

    st.divider()

    # Retrieval Settings
    st.subheader("Retrieval Settings")
    config["retrieval"]["top_k"] = st.slider("Top-K Documents", 1, 10, 4)
    config["retrieval"]["search_type"] = st.selectbox(
        "Search Type", ["hybrid", "similarity", "mmr"]
    )
    config["chunking"]["chunk_size"] = st.slider("Chunk Size", 200, 2000, 1000, 100)
    config["chunking"]["chunk_overlap"] = st.slider("Chunk Overlap", 0, 500, 200, 50)

    st.divider()

    # Document Upload
    st.subheader("Documents")
    supported_types = list(LOADER_MAP.keys())
    uploaded_files = st.file_uploader(
        f"Upload files ({', '.join(supported_types)})",
        accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in supported_types],
    )

    urls_input = st.text_area("Or enter URLs (one per line)", height=80)

    if st.button("Process Documents", type="primary", use_container_width=True):
        process_documents(uploaded_files, urls_input, config)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = ChatHistory()
        st.rerun()


# --- Main Chat Area ---
st.title("DocMind AI")

if not st.session_state.processed:
    st.info(
        "Upload documents in the sidebar and click **Process Documents** to get started."
    )
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.session_state.chat_history.add_user_message(prompt)
                answer = query(
                    st.session_state.chain,
                    prompt,
                    st.session_state.chat_history.get_messages(),
                )
                st.session_state.chat_history.add_ai_message(answer)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
