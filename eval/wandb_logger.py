"""Weights & Biases logging utilities for RAG evaluation."""
from __future__ import annotations

from datetime import datetime, timezone


def init_eval_run(
    method_name: str,
    project: str = "docmind-eval",
    group: str | None = None,
    config: dict | None = None,
):
    """Initialize a W&B run for a single retrieval method evaluation.

    Args:
        method_name: Name of the retrieval method being evaluated.
        project: W&B project name.
        group: Run group (defaults to a timestamp).
        config: Extra config to log.

    Returns:
        The wandb run object.
    """
    import wandb

    if group is None:
        group = datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")

    run = wandb.init(
        project=project,
        group=group,
        name=f"eval-{method_name}",
        config={"retrieval_method": method_name, **(config or {})},
        reinit=True,
    )
    return run


def log_metrics(run, metrics: dict[str, float]) -> None:
    """Log RAGAS metric scores to the run summary.

    Args:
        run: Active wandb run.
        metrics: Dict of metric_name -> score.
    """
    for key, value in metrics.items():
        run.summary[key] = value
    run.summary.update()


def log_per_question_results(
    run,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> None:
    """Log a W&B Table with per-question details.

    Args:
        run: Active wandb run.
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of context lists.
        ground_truths: List of reference answers.
    """
    import wandb

    columns = ["question", "answer", "ground_truth", "num_contexts", "contexts_preview"]
    table = wandb.Table(columns=columns)

    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        preview = " | ".join(c[:100] for c in ctx[:3])
        table.add_data(q, a, gt, len(ctx), preview)

    run.log({"per_question_results": table})


def log_comparison_table(
    run,
    all_results: dict[str, dict[str, float]],
) -> None:
    """Log a summary comparison table and bar charts across methods.

    Args:
        run: Active wandb run (typically the last one or a dedicated summary run).
        all_results: Dict mapping method_name -> {metric: score}.
    """
    import wandb

    if not all_results:
        return

    # Collect all metric names
    metric_names = sorted(
        {k for scores in all_results.values() for k in scores}
    )

    # Summary table
    columns = ["method"] + metric_names
    table = wandb.Table(columns=columns)
    for method, scores in all_results.items():
        row = [method] + [scores.get(m, 0.0) for m in metric_names]
        table.add_data(*row)

    run.log({"comparison_table": table})

    # Bar charts for each metric
    for metric in metric_names:
        data = [[m, scores.get(metric, 0.0)] for m, scores in all_results.items()]
        bar_table = wandb.Table(data=data, columns=["method", metric])
        run.log({
            f"chart_{metric}": wandb.plot.bar(
                bar_table, "method", metric, title=f"{metric} by Method"
            )
        })
