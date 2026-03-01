# DocMind AI

An intelligent document Q&A chatbot powered by LangChain and RAG (Retrieval-Augmented Generation). Upload documents, ask questions, and get accurate answers grounded in your data.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Multi-format document ingestion**: PDF, TXT, CSV, Markdown, Web URLs
- **RAG pipeline**: Semantic search with FAISS vector store
- **Conversational memory**: Follow-up questions with context awareness
- **Hybrid retrieval**: BM25 + vector search with Reciprocal Rank Fusion
- **Streaming responses**: Real-time token-by-token output
- **Source attribution**: Every answer cites its source documents
- **Multiple LLM providers**: OpenAI, Anthropic (Claude), Google (Gemini)
- **Streamlit UI**: Clean, interactive web interface
- **CLI mode**: Command-line interface for scripting

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Contextualize   │  ← Rewrite follow-ups using chat history
│  (if follow-up)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hybrid Retrieval│  ← BM25 + FAISS vector search + RRF
│  (top-k chunks)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate Answer │  ← LLM with retrieved context
│  + Sources       │
└─────────────────┘
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/docmind-ai.git
cd docmind-ai
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

You need at least one LLM provider key:
- `OPENAI_API_KEY` for OpenAI models (default)
- `ANTHROPIC_API_KEY` for Claude models
- `GOOGLE_API_KEY` for Gemini models

### 3. Run the App

```bash
# Streamlit UI
streamlit run app.py

# CLI mode
python -m src.cli --docs data/sample_docs/ --query "What is this document about?"
```

### 4. Use It

1. Upload documents via the sidebar (PDF, TXT, CSV, MD)
2. Or paste a URL to ingest a web page
3. Click "Process Documents" to build the knowledge base
4. Ask questions in the chat input
5. Get answers with source citations

## Configuration

Edit `config.yaml` or use environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai`, `anthropic`, `google` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `4` | Number of chunks to retrieve |
| `TEMPERATURE` | `0` | LLM temperature |

## Project Structure

```
docmind-ai/
├── app.py                 # Streamlit web application
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── src/
│   ├── __init__.py
│   ├── document_loader.py # Multi-format document loading
│   ├── text_splitter.py   # Text chunking with configurable params
│   ├── embeddings.py      # Embedding model management
│   ├── vector_store.py    # FAISS vector store operations
│   ├── retriever.py       # Hybrid retrieval (BM25 + vector + RRF)
│   ├── chain.py           # RAG chain with LCEL
│   ├── memory.py          # Conversation history management
│   ├── config.py          # Configuration management
│   └── cli.py             # Command-line interface
├── tests/
│   ├── test_loader.py
│   ├── test_splitter.py
│   ├── test_retriever.py
│   └── test_chain.py
└── data/
    └── sample_docs/       # Sample documents for testing
```

## License

MIT License. See [LICENSE](LICENSE) for details.
