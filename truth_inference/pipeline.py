from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import pandas as pd

from truth_inference.aggregators import BaseAggregator
from truth_inference.completers import BaseCompleter


def _standardize_answer_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"object", "task", "task_id"}:
            col_map[col] = "object"
        elif lowered in {"worker", "annotator", "user"}:
            col_map[col] = "worker"
        elif lowered in {"response", "label", "answer"}:
            col_map[col] = "response"
    renamed = df.rename(columns=col_map)
    required = ["object", "worker", "response"]
    missing = set(required).difference(renamed.columns)
    if missing and len(renamed.columns) >= 3:
        # Fallback: assume first three columns are object/worker/response.
        fallback_map = {
            renamed.columns[0]: "object",
            renamed.columns[1]: "worker",
            renamed.columns[2]: "response",
        }
        renamed = renamed.rename(columns=fallback_map)
        missing = set(required).difference(renamed.columns)
    if missing:
        raise ValueError(f"Missing required columns in answer data: {sorted(missing)}")
    standardized = renamed[required].copy()
    standardized["object"] = standardized["object"].astype(str).str.strip()
    standardized["worker"] = standardized["worker"].astype(str).str.strip()
    standardized["response"] = standardized["response"].astype(str).str.strip()
    return standardized


def _standardize_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"object", "task", "task_id"}:
            col_map[col] = "object"
        elif lowered in {"truth", "label", "gold"}:
            col_map[col] = "truth"
    renamed = df.rename(columns=col_map)
    required = ["object", "truth"]
    missing = set(required).difference(renamed.columns)
    if missing and len(renamed.columns) >= 2:
        # Fallback: assume first two columns are object and truth.
        fallback_map = {renamed.columns[0]: "object", renamed.columns[1]: "truth"}
        renamed = renamed.rename(columns=fallback_map)
        missing = set(required).difference(renamed.columns)
    if missing:
        raise ValueError(f"Missing required columns in truth data: {sorted(missing)}")
    standardized = renamed[required].copy()
    standardized["object"] = standardized["object"].astype(str).str.strip()
    standardized["truth"] = standardized["truth"].astype(str).str.strip()
    return standardized


def load_dataset(dataset_name: str, data_root: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_root) / dataset_name
    answer_path = base / "answer.csv"
    truth_path = base / "truth.csv"
    if not answer_path.exists():
        raise FileNotFoundError(f"answer.csv not found at {answer_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"truth.csv not found at {truth_path}")
    answers_raw = pd.read_csv(answer_path)
    truth_raw = pd.read_csv(truth_path)
    answers = _standardize_answer_columns(answers_raw)
    truth = _standardize_truth_columns(truth_raw)
    return answers, truth


def compute_accuracy(predictions: pd.Series, truth: pd.DataFrame) -> float:
    truth_series = truth.set_index("object")["truth"]
    common = predictions.index.intersection(truth_series.index)
    if len(common) == 0:
        return float("nan")
    aligned_pred = predictions.loc[common].astype(str)
    aligned_truth = truth_series.loc[common].astype(str)
    return (aligned_pred == aligned_truth).mean()


class TruthInferencePipeline:
    def __init__(self, completer: BaseCompleter, aggregator: BaseAggregator) -> None:
        self.completer = completer
        self.aggregator = aggregator

    def run(
        self,
        answers: pd.DataFrame,
        truth: pd.DataFrame,
        cgmatch_split: bool = False,
        cgmatch_auto_thresholds: bool = False,
        easy_quantile: float = 0.75,
        gap_quantile: float = 0.75,
        easy_confidence: float = 0.8,
        gap_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        print("Starting label completion...", flush=True)
        completed_answers = self.completer.complete(answers)
        print(f"Completion finished. Rows: {len(completed_answers)}. Starting aggregation...", flush=True)
        difficulty_stats = getattr(self.completer, "last_difficulty_stats", None)
        if cgmatch_split:
            base_stats = difficulty_stats if difficulty_stats is not None else self._summarize_confidence_gap(completed_answers)
            if cgmatch_auto_thresholds:
                thresholds = self._auto_thresholds(
                    base_stats,
                    easy_quantile=easy_quantile,
                    gap_quantile=gap_quantile,
                )
                print(
                    "CGMatch auto thresholds -> tau_e={easy:.3f}, tau_a={gap:.3f}".format(
                        easy=thresholds["easy_confidence"],
                        gap=thresholds["gap_threshold"],
                    ),
                    flush=True,
                )
            else:
                thresholds = {
                    "easy_confidence": easy_confidence,
                    "gap_threshold": gap_threshold,
                }
            difficulty_stats = self._compute_difficulty_stats(base_stats, **thresholds)
            counts = difficulty_stats["difficulty"].value_counts().to_dict()
            print(
                "Difficulty buckets: easy={easy}, ambiguous={ambiguous}, hard={hard}".format(
                    easy=counts.get("easy", 0),
                    ambiguous=counts.get("ambiguous", 0),
                    hard=counts.get("hard", 0),
                ),
                flush=True,
            )
            predictions = self._aggregate_by_difficulty(completed_answers, difficulty_stats)
        else:
            predictions = self.aggregator.aggregate(completed_answers)
        print("Aggregation finished. Computing metrics...", flush=True)
        accuracy = compute_accuracy(predictions, truth)
        bucket_accuracy = None
        if difficulty_stats is not None:
            bucket_accuracy = self._compute_bucket_accuracy(predictions, truth, difficulty_stats)
            for name, acc in bucket_accuracy.items():
                print(f"Accuracy ({name}): {acc:.4f}", flush=True)
        return {
            "completed_answers": completed_answers,
            "predictions": predictions,
            "difficulty_stats": difficulty_stats,
            "bucket_accuracy": bucket_accuracy,
            "accuracy": accuracy,
        }

    def _summarize_confidence_gap(self, completed_answers: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for object_id, labels in completed_answers.groupby("object")["response"]:
            counts = labels.value_counts()
            total = counts.sum()
            if total == 0:
                confidence = 0.0
                gap = 0.0
                variability = 0.0
            else:
                top_two = counts.sort_values(ascending=False).tolist()
                top1 = top_two[0] if top_two else 0
                top2 = top_two[1] if len(top_two) > 1 else 0
                confidence = float(top1 / total)
                gap = float((top1 - top2) / total) if total > 0 else 0.0
                probs = counts / total
                variability = float(np.var(probs))
            rows.append(
                {
                    "object": object_id,
                    "confidence": confidence,
                    "count_gap": gap,
                    "variability": variability,
                }
            )
        return pd.DataFrame(rows)

    def _compute_difficulty_stats(
        self,
        base_stats: pd.DataFrame,
        easy_confidence: float,
        gap_threshold: float,
    ) -> pd.DataFrame:
        stats = base_stats.copy()
        stats["difficulty"] = stats.apply(
            lambda row: self._assign_difficulty(
                confidence=row["confidence"],
                count_gap=row["count_gap"],
                easy_confidence=easy_confidence,
                gap_threshold=gap_threshold,
            ),
            axis=1,
        )
        return stats

    def _auto_thresholds(
        self,
        base_stats: pd.DataFrame,
        easy_quantile: float,
        gap_quantile: float,
    ) -> Dict[str, float]:
        if base_stats.empty:
            return {
                "easy_confidence": 1.0,
                "gap_threshold": 0.0,
            }
        confs = base_stats["confidence"]
        return {
            "easy_confidence": float(confs.quantile(easy_quantile)),
            "gap_threshold": float(base_stats["count_gap"].quantile(gap_quantile)),
        }

    def _assign_difficulty(
        self,
        confidence: float,
        count_gap: float,
        easy_confidence: float,
        gap_threshold: float,
    ) -> str:
        if confidence >= easy_confidence:
            return "easy"
        if count_gap >= gap_threshold:
            return "ambiguous"
        return "hard"

    def _aggregate_by_difficulty(self, completed_answers: pd.DataFrame, difficulty_stats: pd.DataFrame) -> pd.Series:
        merged = completed_answers.merge(difficulty_stats[["object", "difficulty"]], on="object", how="left")
        predictions = []
        for difficulty in ["easy", "ambiguous", "hard"]:
            subset = merged.loc[merged["difficulty"] == difficulty, ["object", "worker", "response"]]
            if subset.empty:
                continue
            preds = self.aggregator.aggregate(subset)
            predictions.append(preds)
        if not predictions:
            return pd.Series(dtype=object)
        return pd.concat(predictions)

    def _compute_bucket_accuracy(
        self, predictions: pd.Series, truth: pd.DataFrame, difficulty_stats: pd.DataFrame
    ) -> Dict[str, float]:
        bucket_acc: Dict[str, float] = {}
        for difficulty in ["easy", "ambiguous", "hard"]:
            bucket_objects = difficulty_stats.loc[difficulty_stats["difficulty"] == difficulty, "object"]
            if bucket_objects.empty:
                continue
            preds_bucket = predictions.loc[predictions.index.intersection(bucket_objects)]
            acc = compute_accuracy(preds_bucket, truth)
            bucket_acc[difficulty] = acc
        return bucket_acc
