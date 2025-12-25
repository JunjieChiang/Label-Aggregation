"""
python main_new.py --dataset leaves --completion gcn --aggregater mv --cgmatch-in-iter --cgmatch-auto-thresholds --plot-difficulty --save-results --progress --completion-workers 4 --aggregation-workers 4

python main_new.py --dataset leaves --completion gcn --aggregater mv \
  --cgmatch-in-iter --cgmatch-warmup-steps 40 \
  --warmup-edge-weight 1.0 --object-loss-weight 0.3 --edge-loss-weight 1.0 \
  --reliability-smoothing 0.2 --ambiguous-weight 0.5 --hard-weight 0 --hard-completion object \
  --cgmatch-auto-thresholds --results-dir results_new --plot-difficulty --progress --save-results


# DS
python main_new.py --dataset leaves --completion gcn --aggregater ds

# WSLC
python main_new.py --dataset leaves --completion gcn --aggregater wslc --wslc-embedding-mode manual

# LDPLC
python main_new.py --dataset leaves --completion gcn --aggregater ldplc --ldplc-k 5 --ldplc-iters 5  
"""


from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from truth_inference.aggregators_new import (
    DawidSkeneAggregator,
    LDPLCAggregator,
    MajorityVoteAggregator,
    TiReMGEAggregator,
    WSLCAggregator,
)
from truth_inference.completers_base_new import EMLabelCompletion, NoneCompleter
from truth_inference.completers_new import GCNLabelCompletion
from truth_inference.pipeline_io_new import load_dataset
from truth_inference.pipeline_new import TruthInferencePipeline


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
    if normalized in {"gcn", "tiremge"}:
        completion_auto = args.completion_auto_thresholds or args.cgmatch_auto_thresholds
        completion_easy_quantile = (
            args.completion_easy_quantile if args.completion_auto_thresholds else args.easy_quantile
        )
        completion_gap_quantile = (
            args.completion_gap_quantile if args.completion_auto_thresholds else args.gap_quantile
        )
        return GCNLabelCompletion(
            max_steps=args.gcn_steps,
            learning_rate=args.gcn_learning_rate,
            show_progress=args.progress,
            seed=args.seed,
            cgmatch_in_completion=args.cgmatch_in_iter,
            cgmatch_momentum=args.cgmatch_momentum,
            cgmatch_warmup_steps=args.cgmatch_warmup_steps,
            ambiguous_weight=args.ambiguous_weight,
            hard_weight=args.hard_weight,
            reliability_alpha=args.reliability_alpha,
            reliability_ema=args.reliability_ema,
            reliability_smoothing=args.reliability_smoothing,
            reliability_temperature=args.reliability_temperature,
            object_loss_weight=args.object_loss_weight,
            edge_loss_weight=args.edge_loss_weight,
            warmup_edge_weight=args.warmup_edge_weight,
            hidden_dim=args.gcn_hidden_dim,
            dropout=args.gcn_dropout,
            device=args.device,
            hard_completion=args.hard_completion,
            completion_easy_confidence=args.completion_easy_confidence,
            completion_gap_threshold=args.completion_gap_threshold,
            use_cgmatch_thresholds=not args.completion_ignore_cgmatch_thresholds,
            completion_auto_thresholds=completion_auto,
            completion_easy_quantile=completion_easy_quantile,
            completion_gap_quantile=completion_gap_quantile,
        )
    raise ValueError(f"Unsupported completion algorithm: {name}")


def _build_aggregator(name: str, args: argparse.Namespace):
    normalized = name.lower()
    if normalized == "mv":
        return MajorityVoteAggregator(num_workers=args.aggregation_workers, show_progress=args.progress)
    if normalized == "ds":
        return DawidSkeneAggregator(
            max_iters=args.ds_iters,
            smoothing=args.ds_smoothing,
            convergence_tol=args.ds_tol,
            show_progress=args.progress,
        )
    if normalized == "tiremge":
        return TiReMGEAggregator(
            max_steps=args.tiremge_steps,
            learning_rate=args.tiremge_learning_rate,
            show_progress=args.progress,
            seed=args.seed,
            cgmatch_stats=args.cgmatch_in_iter,
        )
    if normalized == "wslc":
        return WSLCAggregator(
            num_neighbors=args.wslc_neighbors,
            embedding_mode=args.wslc_embedding_mode,
            graphsage_dim=args.wslc_graphsage_dim,
            graphsage_layers=args.wslc_graphsage_layers,
        )
    if normalized == "ldplc":
        return LDPLCAggregator(
            k=args.ldplc_k,
            iterations=args.ldplc_iters,
            num_threads=args.ldplc_threads,
        )
    raise ValueError(f"Unsupported aggregation algorithm: {name}")


def _save_outputs(
    dataset: str,
    output_root: Path,
    completed_answers: pd.DataFrame,
    predictions: pd.Series,
    metrics: Dict[str, float | Dict[str, float] | str],
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
    aggregator = _build_aggregator(args.aggregater, args)
    pipeline = TruthInferencePipeline(completer, aggregator)
    result = pipeline.run(
        answers,
        truth,
        cgmatch_auto_thresholds=args.cgmatch_auto_thresholds,
        easy_quantile=args.easy_quantile,
        gap_quantile=args.gap_quantile,
        easy_confidence=args.easy_confidence,
        gap_threshold=args.gap_threshold,
    )

    metrics_payload: Dict[str, object] = {
        "dataset": args.dataset,
        "completion": args.completion,
        "aggregater": args.aggregater,
        "before_accuracy": result["before_accuracy"],
        "full_accuracy": result["full_accuracy"],
        "bucket_accuracy": result["bucket_accuracy"],
    }

    print(f"Dataset: {args.dataset}")
    print(f"Completion: {args.completion}")
    print(f"Aggregater: {args.aggregater}")
    print(f"Accuracy (before): {result['before_accuracy']:.4f}")
    print(f"Accuracy (full): {result['full_accuracy']:.4f}")
    for name, acc in result["bucket_accuracy"].items():
        print(f"Accuracy ({name}): {acc:.4f}")
    if result.get("difficulty_stats") is not None:
        counts = result["difficulty_stats"]["difficulty"].value_counts().to_dict()
        metrics_payload["difficulty_split_counts"] = counts
        print(f"Difficulty split counts: {counts}")

    if args.save_results:
        if args.plot_difficulty and result.get("difficulty_stats") is not None:
            _plot_difficulty_scatter(args.dataset, Path(args.results_dir), result["difficulty_stats"])
        _save_outputs(
            args.dataset,
            Path(args.results_dir),
            result["completed_answers"],
            result["full_predictions"],
            metrics_payload,
            difficulty_stats=result.get("difficulty_stats"),
        )
        print(f"Saved outputs to {Path(args.results_dir) / args.dataset}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Truth inference with CGMatch-style completion and aggregation.")
    parser.add_argument("--dataset", help="Dataset name (expects data/<dataset>/answer.csv and truth.csv)")
    parser.add_argument("--data-root", default="data", help="Root directory containing dataset subfolders.")
    parser.add_argument(
        "--completion",
        default="none",
        choices=["none", "em", "gcn", "tiremge"],
        help="Label completion algorithm to use (gcn is the PyG implementation).",
    )
    parser.add_argument(
        "--aggregater",
        "--aggregation",
        dest="aggregater",
        default="mv",
        choices=["mv", "tiremge", "ds", "wslc", "ldplc"],
        help="Aggregation algorithm to use after completion.",
    )
    parser.add_argument("--em-iters", type=int, default=100, help="Maximum EM iterations for em completion.")
    parser.add_argument("--em-smoothing", type=float, default=1e-2, help="Laplace smoothing for worker confusion matrices.")
    parser.add_argument("--em-tol", type=float, default=1e-5, help="Convergence tolerance on posteriors for em completion.")
    parser.add_argument(
        "--gcn-steps",
        type=int,
        default=200,
        help="Training iterations for GCN completion (used with --completion gcn).",
    )
    parser.add_argument(
        "--gcn-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for GCN completion (used with --completion gcn).",
    )
    parser.add_argument(
        "--object-loss-weight",
        type=float,
        default=1.0,
        help="Weight for object-level consensus loss during GCN completion.",
    )
    parser.add_argument(
        "--edge-loss-weight",
        type=float,
        default=1.0,
        help="Weight for edge-level loss during GCN completion (after warmup).",
    )
    parser.add_argument(
        "--warmup-edge-weight",
        type=float,
        default=0.0,
        help="Edge loss weight during warmup steps.",
    )
    parser.add_argument(
        "--tiremge-steps",
        type=int,
        default=200,
        help="Training iterations for TiReMGE aggregation (used with --aggregater tiremge).",
    )
    parser.add_argument(
        "--tiremge-learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate for TiReMGE aggregation (used with --aggregater tiremge).",
    )
    parser.add_argument(
        "--ds-iters",
        type=int,
        default=50,
        help="Maximum iterations for Dawid-Skene aggregation.",
    )
    parser.add_argument(
        "--ds-smoothing",
        type=float,
        default=1e-2,
        help="Laplace smoothing for Dawid-Skene confusion matrices.",
    )
    parser.add_argument(
        "--ds-tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance for Dawid-Skene aggregation.",
    )
    parser.add_argument(
        "--wslc-neighbors",
        type=int,
        default=None,
        help="Number of neighbors for WSLC worker similarity (default: all).",
    )
    parser.add_argument(
        "--wslc-embedding-mode",
        choices=["manual", "graphsage"],
        default="manual",
        help="Worker feature construction for WSLC.",
    )
    parser.add_argument(
        "--wslc-graphsage-dim",
        type=int,
        default=None,
        help="Embedding dimension for WSLC GraphSAGE worker vectors.",
    )
    parser.add_argument(
        "--wslc-graphsage-layers",
        type=int,
        default=2,
        help="Number of GraphSAGE layers for WSLC.",
    )
    parser.add_argument(
        "--ldplc-k",
        type=int,
        default=5,
        help="Number of nearest neighbors for LDPLC.",
    )
    parser.add_argument(
        "--ldplc-iters",
        type=int,
        default=5,
        help="Propagation iterations for LDPLC.",
    )
    parser.add_argument(
        "--ldplc-threads",
        type=int,
        default=1,
        help="Number of worker threads for LDPLC.",
    )
    parser.add_argument(
        "--gcn-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for the GCN encoder.",
    )
    parser.add_argument(
        "--gcn-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the GCN encoder.",
    )
    parser.add_argument(
        "--reliability-alpha",
        type=float,
        default=1.0,
        help="Scale factor for reliability updates (exp(-alpha * loss)).",
    )
    parser.add_argument(
        "--reliability-ema",
        type=float,
        default=0.9,
        help="EMA factor for reliability updates.",
    )
    parser.add_argument(
        "--reliability-smoothing",
        type=float,
        default=1.0,
        help="Mix factor for source-source reliability smoothing (0 disables).",
    )
    parser.add_argument(
        "--reliability-temperature",
        type=float,
        default=0.5,
        help="Temperature for source-source similarity softmax.",
    )
    parser.add_argument(
        "--cgmatch-warmup-steps",
        type=int,
        default=40,
        help="Warmup steps with fixed reliability before CGMatch weighting.",
    )
    parser.add_argument(
        "--ambiguous-weight",
        type=float,
        default=0.5,
        help="Reliability weight applied to ambiguous objects when updating sources.",
    )
    parser.add_argument(
        "--hard-weight",
        type=float,
        default=0.0,
        help="Reliability weight applied to hard objects when updating sources.",
    )
    parser.add_argument(
        "--hard-completion",
        choices=["skip", "object", "pairwise"],
        default="skip",
        help="How to complete hard objects: skip, use object head, or pairwise head.",
    )
    parser.add_argument(
        "--completion-easy-confidence",
        type=float,
        default=0.8,
        help="Confidence threshold for completion difficulty buckets.",
    )
    parser.add_argument(
        "--completion-gap-threshold",
        type=float,
        default=0.15,
        help="Count-gap threshold for completion difficulty buckets.",
    )
    parser.add_argument(
        "--completion-auto-thresholds",
        action="store_true",
        help="Use completion quantiles to set difficulty thresholds during completion.",
    )
    parser.add_argument(
        "--completion-easy-quantile",
        type=float,
        default=0.75,
        help="Quantile of completion confidence used as tau_e.",
    )
    parser.add_argument(
        "--completion-gap-quantile",
        type=float,
        default=0.75,
        help="Quantile of completion count-gap used as tau_a.",
    )
    parser.add_argument(
        "--completion-ignore-cgmatch-thresholds",
        action="store_true",
        help="Use fixed completion thresholds instead of CGMatch EMA thresholds.",
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
        default=4,
        help="Number of worker threads for label completion (imputation phase).",
    )
    parser.add_argument(
        "--aggregation-workers",
        type=int,
        default=4,
        help="Number of worker threads for aggregation.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for GCN completion (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--cgmatch-in-iter",
        action="store_true",
        help="Enable CGMatch-style reliability weighting during completion.",
    )
    parser.add_argument(
        "--cgmatch-momentum",
        type=float,
        default=0.999,
        help="Momentum for EMA thresholds (tau_e, tau_a) inside completion.",
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
    parser.add_argument("--results-dir", default="results/gcn_new", help="Directory root used when saving outputs.")
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
        if args.aggregater == "tiremge":
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
