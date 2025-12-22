"""
python main.py --dataset leaves --completion tiremge --cgmatch-in-iter --cgmatch-split --cgmatch-auto-thresholds --plot-difficulty --save-results --progress --completion-workers 4 --aggregation-workers 4

"""


from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from truth_inference.aggregators import MajorityVoteAggregator, TiReMGEAggregator
from truth_inference.completers import EMLabelCompletion, NoneCompleter, TiReMGELabelCompletion
from truth_inference.pipeline import TruthInferencePipeline, load_dataset, compute_accuracy


def _build_completer(name: str, args: argparse.Namespace):
    normalized = name.lower()
    if normalized == "none":
        return NoneCompleter()
    if normalized == "em":
        return EMLabelCompletion(
            max_iters=args.em_iters,
            smoothing=args.em_smoothing,
            convergence_tol=args.em_tol,
            num_workers=args.completion_workers,
            show_progress=args.progress,
            cgmatch_in_em=args.cgmatch_in_iter,
            cgmatch_momentum=args.cgmatch_momentum,
        )
    if normalized == "tiremge":
        return TiReMGELabelCompletion(
            max_steps=args.tiremge_steps,
            learning_rate=args.tiremge_learning_rate,
            show_progress=args.progress,
            seed=args.seed,
            cgmatch_in_completion=args.cgmatch_in_iter,
        )
    raise ValueError(f"Unsupported completion algorithm: {name}")


def _build_aggregator(name: str, args: argparse.Namespace):
    normalized = name.lower()
    if normalized == "mv":
        return MajorityVoteAggregator(num_workers=args.aggregation_workers, show_progress=args.progress)
    if normalized == "tiremge":
        return TiReMGEAggregator(
            max_steps=args.tiremge_steps,
            learning_rate=args.tiremge_learning_rate,
            show_progress=args.progress,
            seed=args.seed,
            cgmatch_stats=args.cgmatch_in_iter,
        )
    raise ValueError(f"Unsupported aggregation algorithm: {name}")


def _save_outputs(
    dataset: str,
    output_root: Path,
    completed_answers: pd.DataFrame,
    predictions: pd.Series,
    metrics: Dict[str, float],
    difficulty_stats: pd.DataFrame | None = None,
) -> None:
    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_answers.to_csv(output_dir / "completed_answers.csv", index=False)
    pred_df = predictions.reset_index()
    pred_df.columns = ["object", "prediction"]
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    if difficulty_stats is not None:
        difficulty_stats.to_csv(output_dir / "difficulty_stats.csv", index=False)
        pred_with_difficulty = pred_df.merge(difficulty_stats[["object", "difficulty"]], on="object", how="left")
        pred_with_difficulty.to_csv(output_dir / "predictions_with_difficulty.csv", index=False)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def _plot_difficulty_scatter(
    dataset: str, output_root: Path, difficulty_stats: pd.DataFrame, show: bool = False
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping difficulty plot.", flush=True)
        return

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {"easy": "gold", "ambiguous": "dodgerblue", "hard": "purple"}
    if "variability" not in difficulty_stats.columns:
        print("variability not found in difficulty_stats; skipping difficulty plot.", flush=True)
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    scatter = ax.scatter(
        difficulty_stats["variability"],
        difficulty_stats["confidence"],
        c=difficulty_stats["count_gap"],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    ax.set_xlabel("Variability")
    ax.set_ylabel("Confidence")
    ax.set_title("CGMatch difficulty map")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Count-Gap")
    fig.tight_layout()
    path = output_dir / "difficulty_map.png"
    fig.savefig(path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved difficulty scatter to {path}", flush=True)


def _run_pipeline(args: argparse.Namespace) -> None:
    answers, truth = load_dataset(args.dataset, data_root=args.data_root)
    completer = _build_completer(args.completion, args)
    aggregator = _build_aggregator(args.aggregation, args)
    pipeline = TruthInferencePipeline(completer, aggregator)
    result = pipeline.run(
        answers,
        truth,
        cgmatch_split=args.cgmatch_split,
        cgmatch_auto_thresholds=args.cgmatch_auto_thresholds,
        easy_quantile=args.easy_quantile,
        gap_quantile=args.gap_quantile,
        easy_confidence=args.easy_confidence,
        gap_threshold=args.gap_threshold,
    )

    metrics_payload: Dict[str, object] = {
        "dataset": args.dataset,
        "completion": args.completion,
        "aggregation": args.aggregation,
        "accuracy": result["accuracy"],
    }

    print(f"Dataset: {args.dataset}")
    print(f"Completion: {args.completion}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    if args.cgmatch_split and result.get("difficulty_stats") is not None:
        counts = result["difficulty_stats"]["difficulty"].value_counts().to_dict()
        metrics_payload["difficulty_split_counts"] = counts
        print(f"Difficulty split counts: {counts}")
        if result.get("bucket_accuracy"):
            metrics_payload["bucket_accuracy"] = result["bucket_accuracy"]
            for name, acc in result["bucket_accuracy"].items():
                print(f"Accuracy ({name}): {acc:.4f}")

    if args.save_results:
        if args.plot_difficulty and result.get("difficulty_stats") is not None:
            _plot_difficulty_scatter(args.dataset, Path(args.results_dir), result["difficulty_stats"])
        _save_outputs(
            args.dataset,
            Path(args.results_dir),
            result["completed_answers"],
            result["predictions"],
            metrics_payload,
            difficulty_stats=result.get("difficulty_stats"),
        )
        print(f"Saved outputs to {Path(args.results_dir) / args.dataset}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Truth inference with pluggable label completion and aggregation.")
    parser.add_argument("--dataset", help="Dataset name (expects data/<dataset>/answer.csv and truth.csv)")
    parser.add_argument("--data-root", default="data", help="Root directory containing dataset subfolders.")
    parser.add_argument(
        "--completion",
        default="none",
        choices=["none", "em", "tiremge"],
        help="Label completion algorithm to use.",
    )
    parser.add_argument(
        "--aggregation",
        default="mv",
        choices=["mv", "tiremge"],
        help="Aggregation algorithm to use after completion.",
    )
    parser.add_argument("--em-iters", type=int, default=100, help="Maximum EM iterations for em completion.")
    parser.add_argument("--em-smoothing", type=float, default=1e-2, help="Laplace smoothing for worker confusion matrices.")
    parser.add_argument("--em-tol", type=float, default=1e-5, help="Convergence tolerance on posteriors for em completion.")
    parser.add_argument(
        "--tiremge-steps",
        type=int,
        default=200,
        help="Training iterations for the TiReMGE graph completion (used with --completion tiremge).",
    )
    parser.add_argument(
        "--tiremge-learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate for TiReMGE (used with --completion tiremge).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--completion-workers",
        type=int,
        default=1,
        help="Number of worker threads for label completion (imputation phase).",
    )
    parser.add_argument(
        "--aggregation-workers",
        type=int,
        default=1,
        help="Number of worker threads for aggregation.",
    )
    parser.add_argument(
        "--cgmatch-in-iter",
        action="store_true",
        help="Enable CGMatch-style stats during completion (EM or TiReMGE) for downstream difficulty splitting.",
    )
    parser.add_argument(
        "--cgmatch-momentum",
        type=float,
        default=0.999,
        help="Momentum for EMA thresholds (tau_e, tau_a) inside EM when CGMatch is enabled.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bars during completion/aggregation (requires tqdm).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, save completed answers, predictions, and metrics to results/<dataset>/.",
    )
    parser.add_argument("--results-dir", default="results/tiremge", help="Directory root used when saving outputs.")
    parser.add_argument(
        "--cgmatch-split",
        action="store_true",
        help="Use CGMatch-style count-gap/confidence split (easy/ambiguous/hard) before aggregation.",
    )
    parser.add_argument(
        "--cgmatch-auto-thresholds",
        action="store_true",
        help="Use dataset quantiles to set CGMatch thresholds (tau_e, tau_a) instead of fixed values.",
    )
    parser.add_argument(
        "--easy-quantile",
        type=float,
        default=0.75,
        help="Quantile of confidence used as tau_e when auto thresholds are enabled.",
    )
    parser.add_argument(
        "--gap-quantile",
        type=float,
        default=0.75,
        help="Quantile of count-gap used as tau_a when auto thresholds are enabled.",
    )
    parser.add_argument(
        "--easy-confidence",
        type=float,
        default=0.8,
        help="tau_e: confidence threshold for 'easy to infer' bucket (max label probability).",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.15,
        help="tau_a: count-gap threshold for ambiguous bucket.",
    )
    parser.add_argument(
        "--plot-difficulty",
        action="store_true",
        help="Save a scatter plot of count-gap vs confidence colored by difficulty buckets.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.dataset:
        parser.error("Please provide --dataset.")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import tensorflow as tf
        except ImportError:
            tf = None
        if tf is not None:
            tf.random.set_seed(args.seed)
            try:
                tf.config.experimental.enable_op_determinism(True)
            except (AttributeError, TypeError):
                pass
    _run_pipeline(args)


if __name__ == "__main__":
    main()
