from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from truth_inference.aggregators_new import BaseAggregator
from truth_inference.completers_base_new import BaseCompleter
from truth_inference.pipeline_io_new import compute_accuracy


class TruthInferencePipeline:
    def __init__(self, completer: BaseCompleter, aggregator: BaseAggregator) -> None:
        self.completer = completer
        self.aggregator = aggregator

    def run(
        self,
        answers: pd.DataFrame,
        truth: pd.DataFrame,
        cgmatch_auto_thresholds: bool = False,
        easy_quantile: float = 0.75,
        gap_quantile: float = 0.75,
        easy_confidence: float = 0.8,
        gap_threshold: float = 0.15,
    ) -> Dict[str, Any]:
        observed_answers = answers[["object", "worker", "response"]].dropna().copy()
        before_predictions = self.aggregator.aggregate(observed_answers)
        before_accuracy = compute_accuracy(before_predictions, truth)
        print("Starting label completion...", flush=True)
        completed_answers = self.completer.complete(answers)
        print(f"Completion finished. Rows: {len(completed_answers)}. Starting aggregation...", flush=True)

        base_stats = self._get_base_stats(completed_answers)
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

        bucket_predictions: Dict[str, pd.Series] = {}
        bucket_accuracy: Dict[str, float] = {}
        merged = completed_answers.merge(difficulty_stats[["object", "difficulty"]], on="object", how="left")
        for difficulty in ["easy", "ambiguous", "hard"]:
            subset = merged.loc[merged["difficulty"] == difficulty, ["object", "worker", "response"]]
            if subset.empty:
                continue
            preds = self.aggregator.aggregate(subset)
            bucket_predictions[difficulty] = preds
            bucket_accuracy[difficulty] = compute_accuracy(preds, truth)

        print("Observed rows:", len(answers), "Completed rows:", len(completed_answers))
        obs_pairs = set(zip(answers["object"], answers["worker"]))
        comp_pairs = set(zip(completed_answers["object"], completed_answers["worker"]))
        print("Added pairs:", len(comp_pairs - obs_pairs))

        full_predictions = self.aggregator.aggregate(completed_answers)
        full_accuracy = compute_accuracy(full_predictions, truth)

        print("Aggregation finished. Computing metrics...", flush=True)

        return {
            "completed_answers": completed_answers,
            "full_predictions": full_predictions,
            "difficulty_stats": difficulty_stats,
            "bucket_predictions": bucket_predictions,
            "bucket_accuracy": bucket_accuracy,
            "full_accuracy": full_accuracy,
            "before_accuracy": before_accuracy,
        }

    def _get_base_stats(self, completed_answers: pd.DataFrame) -> pd.DataFrame:
        difficulty_stats = getattr(self.completer, "last_difficulty_stats", None)
        if difficulty_stats is not None and {"confidence", "count_gap"}.issubset(difficulty_stats.columns):
            columns = ["object", "confidence", "count_gap"]
            if "variability" in difficulty_stats.columns:
                columns.append("variability")
            return difficulty_stats[columns].copy()
        return self._summarize_confidence_gap(completed_answers)

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
        if not rows:
            return pd.DataFrame(columns=["object", "confidence", "count_gap", "variability"])
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
