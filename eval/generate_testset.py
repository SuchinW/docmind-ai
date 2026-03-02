"""Synthetic Q&A test-set generation for RAG evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


def _generate_with_ragas(
    documents: list[Document],
    llm: BaseChatModel,
    num_questions: int,
) -> list[dict]:
    """Try RAGAS TestsetGenerator (may change across versions)."""
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator(llm=llm)
    testset = generator.generate_with_langchain_docs(
        documents, testset_size=num_questions
    )
    # Convert to list of dicts
    rows = []
    for item in testset.to_list():
        rows.append({
            "question": item.get("user_input") or item.get("question", ""),
            "ground_truth": item.get("reference") or item.get("ground_truth", ""),
            "contexts": item.get("reference_contexts") or item.get("contexts", []),
        })
    return rows


def _generate_with_llm_fallback(
    documents: list[Document],
    llm: BaseChatModel,
    num_questions: int,
) -> list[dict]:
    """Fallback: use the LLM directly to generate Q&A pairs."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Combine document text (truncate if very long)
    combined = "\n\n---\n\n".join(
        doc.page_content[:2000] for doc in documents[:10]
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a test-set generator for RAG evaluation. "
            "Given the following document content, generate exactly "
            "{num_questions} question-answer pairs. Each pair should test "
            "whether a RAG system can find and use the right information.\n\n"
            "Return ONLY a valid JSON array. Each element must have:\n"
            '  "question": the question string\n'
            '  "ground_truth": the correct answer based on the documents\n\n'
            "Make questions diverse: factual, comparative, and analytical.\n\n"
            "Documents:\n{documents}",
        ),
        ("human", "Generate {num_questions} Q&A pairs as a JSON array."),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "num_questions": num_questions,
        "documents": combined,
    })

    # Parse JSON from the response (strip markdown fences if present)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    rows = json.loads(raw)
    for row in rows:
        row.setdefault("contexts", [])
    return rows


def generate_testset(
    documents: list[Document],
    llm: BaseChatModel,
    num_questions: int = 10,
) -> list[dict]:
    """Generate a synthetic Q&A testset from documents.

    Tries the RAGAS TestsetGenerator first; falls back to a direct LLM
    prompt if the RAGAS API has changed or is unavailable.

    Returns:
        List of dicts with "question", "ground_truth", and "contexts".
    """
    try:
        return _generate_with_ragas(documents, llm, num_questions)
    except Exception:
        return _generate_with_llm_fallback(documents, llm, num_questions)


def save_testset(testset: list[dict], path: str | Path) -> None:
    """Save a testset to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(testset, f, indent=2)


def load_testset(path: str | Path) -> list[dict]:
    """Load a testset from a JSON file."""
    with open(path) as f:
        return json.load(f)
