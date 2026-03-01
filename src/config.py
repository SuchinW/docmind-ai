"""Configuration management for DocMind AI."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    """Load configuration from config.yaml with environment variable overrides."""
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    if os.getenv("LLM_PROVIDER"):
        config["llm"]["provider"] = os.getenv("LLM_PROVIDER")
    if os.getenv("LLM_MODEL"):
        config["llm"]["model"] = os.getenv("LLM_MODEL")
    if os.getenv("EMBEDDING_MODEL"):
        config["embeddings"]["model"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("CHUNK_SIZE"):
        config["chunking"]["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
    if os.getenv("CHUNK_OVERLAP"):
        config["chunking"]["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
    if os.getenv("TOP_K"):
        config["retrieval"]["top_k"] = int(os.getenv("TOP_K"))
    if os.getenv("TEMPERATURE"):
        config["llm"]["temperature"] = float(os.getenv("TEMPERATURE"))

    return config
