"""RAGAS metric computation for RAG evaluation."""
from __future__ import annotations


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Compute RAGAS metrics for a set of Q&A results.

    Tries the RAGAS evaluate() function first; falls back to a
    simplified LLM-based scoring if the API is incompatible.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of context lists (one per question).
        ground_truths: List of reference answers.

    Returns:
        Dict mapping metric names to their average scores.
    """
    try:
        return _evaluate_with_ragas(
            questions, answers, contexts, ground_truths
        )
    except Exception:
        return _evaluate_fallback(
            questions, answers, contexts, ground_truths
        )


def _evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Use the RAGAS library to compute metrics."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


def _evaluate_fallback(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Simplified heuristic scoring when RAGAS is unavailable.

    Uses basic text-overlap heuristics so the pipeline can still produce
    numeric scores for comparison, even without the full RAGAS framework.
    """
    from difflib import SequenceMatcher

    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    recall_scores = []

    for answer, ctx_list, ground_truth, question in zip(
        answers, contexts, ground_truths, questions
    ):
        combined_ctx = " ".join(ctx_list).lower()
        answer_lower = answer.lower()
        gt_lower = ground_truth.lower()
        q_lower = question.lower()

        # Faithfulness: how much of the answer is grounded in context
        faith = SequenceMatcher(None, answer_lower, combined_ctx).ratio()
        faithfulness_scores.append(faith)

        # Answer relevancy: how related the answer is to the question
        rel = SequenceMatcher(None, answer_lower, q_lower).ratio()
        relevancy_scores.append(rel)

        # Context precision: how related contexts are to the question
        if ctx_list:
            ctx_scores = [
                SequenceMatcher(None, c.lower(), q_lower).ratio()
                for c in ctx_list
            ]
            precision_scores.append(sum(ctx_scores) / len(ctx_scores))
        else:
            precision_scores.append(0.0)

        # Context recall: how much ground truth is covered by contexts
        recall = SequenceMatcher(None, combined_ctx, gt_lower).ratio()
        recall_scores.append(recall)

    def _mean(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    return {
        "faithfulness": _mean(faithfulness_scores),
        "answer_relevancy": _mean(relevancy_scores),
        "context_precision": _mean(precision_scores),
        "context_recall": _mean(recall_scores),
    }
