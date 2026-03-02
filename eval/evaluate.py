"""Main evaluation script for DocMind AI RAG pipeline.

Usage:
    python -m eval.evaluate --docs data/sample_docs/ --num-questions 15
    python -m eval.evaluate --docs data/sample_docs/ --methods similarity hybrid rerank
    python -m eval.evaluate --docs data/sample_docs/ --no-wandb
    python -m eval.evaluate --testset eval/testset/my_testset.json --docs data/sample_docs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.documents import Document


def _load_docs(docs_dir: str) -> list[Document]:
    """Load all .txt files from a directory."""
    from src.document_loader import load_file

    docs_path = Path(docs_dir)
    all_docs: list[Document] = []
    for f in sorted(docs_path.glob("*.txt")):
        all_docs.extend(load_file(f))
    if not all_docs:
        print(f"No .txt documents found in {docs_dir}")
        sys.exit(1)
    return all_docs


def _print_comparison(all_results: dict[str, dict[str, float]]) -> None:
    """Pretty-print a comparison table to the terminal."""
    if not all_results:
        return

    metric_names = sorted(
        {k for scores in all_results.values() for k in scores}
    )

    # Header
    method_col_width = max(len(m) for m in all_results) + 2
    header = "Method".ljust(method_col_width)
    for metric in metric_names:
        header += metric.rjust(20)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for method, scores in all_results.items():
        row = method.ljust(method_col_width)
        for metric in metric_names:
            val = scores.get(metric, 0.0)
            row += f"{val:.4f}".rjust(20)
        print(row)

    print("=" * len(header))

    # Highlight best per metric
    print("\nBest per metric:")
    for metric in metric_names:
        best_method = max(all_results, key=lambda m: all_results[m].get(metric, 0.0))
        best_val = all_results[best_method].get(metric, 0.0)
        print(f"  {metric}: {best_method} ({best_val:.4f})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocMind AI retrieval methods with RAGAS"
    )
    parser.add_argument(
        "--docs", required=True,
        help="Directory containing .txt documents for evaluation",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Retrieval methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--testset", default=None,
        help="Path to a pre-generated testset JSON file",
    )
    parser.add_argument(
        "--num-questions", type=int, default=10,
        help="Number of test questions to generate (default: 10)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project", default="docmind-eval",
        help="W&B project name (default: docmind-eval)",
    )
    args = parser.parse_args()

    # --- Imports (heavy deps loaded after arg parsing for fast --help) ---
    from src.config import load_config
    from src.chain import query_with_contexts, get_llm
    from src.text_splitter import split_documents
    from src.embeddings import get_embeddings
    from src.vector_store import create_vector_store
    from src.retriever import create_retriever, RETRIEVAL_METHODS
    from eval.generate_testset import generate_testset, save_testset, load_testset
    from eval.ragas_metrics import evaluate_ragas

    config = load_config()
    llm = get_llm(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
    )

    # --- Load documents ---
    print(f"Loading documents from {args.docs}...")
    documents = _load_docs(args.docs)
    print(f"  Loaded {len(documents)} document(s)")

    # --- Chunk & embed ---
    print("Splitting documents into chunks...")
    chunks = split_documents(
        documents,
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    print(f"  Created {len(chunks)} chunks")

    print("Creating embeddings and vector store...")
    embeddings = get_embeddings(model=config["embeddings"]["model"])
    vector_store = create_vector_store(chunks, embeddings)
    print("  Vector store ready")

    # --- Generate or load testset ---
    if args.testset and Path(args.testset).exists():
        print(f"Loading testset from {args.testset}...")
        testset = load_testset(args.testset)
    else:
        print(f"Generating {args.num_questions} test questions...")
        testset = generate_testset(documents, llm, args.num_questions)
        save_path = Path("eval/testset/testset.json")
        save_testset(testset, save_path)
        print(f"  Saved testset to {save_path}")

    print(f"  Testset has {len(testset)} questions")

    # --- Determine methods to evaluate ---
    methods = args.methods or RETRIEVAL_METHODS
    invalid = set(methods) - set(RETRIEVAL_METHODS)
    if invalid:
        print(f"Unknown methods: {invalid}. Valid: {RETRIEVAL_METHODS}")
        sys.exit(1)

    # --- W&B setup ---
    use_wandb = not args.no_wandb
    wandb_group = None
    if use_wandb:
        from datetime import datetime, timezone
        wandb_group = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
        print(f"W&B project: {args.wandb_project}, group: {wandb_group}")

    # --- Evaluate each method ---
    all_results: dict[str, dict[str, float]] = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method}")
        print(f"{'='*60}")

        try:
            retriever = create_retriever(
                vector_store=vector_store,
                chunks=chunks,
                top_k=config["retrieval"]["top_k"],
                search_type=method,
                llm=llm,
                original_docs=documents,
            )
        except Exception as e:
            print(f"  Skipping {method}: {e}")
            continue

        # Run each test question
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, item in enumerate(testset):
            q = item["question"]
            gt = item.get("ground_truth", "")
            print(f"  Q{i+1}/{len(testset)}: {q[:80]}...")

            try:
                result = query_with_contexts(retriever, llm, q)
                questions.append(result["question"])
                answers.append(result["answer"])
                contexts.append(result["contexts"])
                ground_truths.append(gt)
            except Exception as e:
                print(f"    Error: {e}")
                continue

        if not questions:
            print(f"  No results for {method}, skipping metrics.")
            continue

        # Compute RAGAS metrics
        print(f"  Computing metrics for {len(questions)} questions...")
        metrics = evaluate_ragas(questions, answers, contexts, ground_truths)
        all_results[method] = metrics

        for metric, score in metrics.items():
            print(f"    {metric}: {score:.4f}")

        # W&B logging
        if use_wandb:
            from eval.wandb_logger import (
                init_eval_run,
                log_metrics,
                log_per_question_results,
            )

            run = init_eval_run(
                method_name=method,
                project=args.wandb_project,
                group=wandb_group,
                config={
                    "num_questions": len(questions),
                    "top_k": config["retrieval"]["top_k"],
                    "chunk_size": config["chunking"]["chunk_size"],
                    "llm_model": config["llm"]["model"],
                },
            )
            log_metrics(run, metrics)
            log_per_question_results(
                run, questions, answers, contexts, ground_truths
            )
            run.finish()

    # --- Comparison ---
    _print_comparison(all_results)

    # Log comparison table to W&B
    if use_wandb and all_results:
        from eval.wandb_logger import init_eval_run, log_comparison_table

        run = init_eval_run(
            method_name="comparison",
            project=args.wandb_project,
            group=wandb_group,
        )
        log_comparison_table(run, all_results)
        run.finish()
        print(f"Results logged to W&B project '{args.wandb_project}'")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
