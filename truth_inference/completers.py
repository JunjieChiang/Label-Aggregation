from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from truth_inference.tiremge import build_difficulty_stats, run_tiremge_prediction

try:  # Optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional
    tqdm = None


class BaseCompleter(ABC):
    @abstractmethod
    def complete(self, answers: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in missing worker-task labels.

        Args:
            answers: Sparse DataFrame with columns ['object', 'worker', 'response'].
        Returns:
            DataFrame with the same columns where (optionally) every (object, worker)
            pair has a response.
        """

    @staticmethod
    def _validate_answers(answers: pd.DataFrame) -> None:
        required = {"object", "worker", "response"}
        missing = required.difference(answers.columns)
        if missing:
            raise ValueError(f"Missing required columns in answers: {sorted(missing)}")


class NoneCompleter(BaseCompleter):
    """Pass-through baseline: no label completion is performed."""

    def complete(self, answers: pd.DataFrame) -> pd.DataFrame:
        self._validate_answers(answers)
        return answers.copy()


class EMLabelCompletion(BaseCompleter):
    """
    Simple EM-style label completion using worker confusion matrices.

    Labels are treated as categorical; worker reliability is captured by a confusion
    matrix theta_j[true, observed]. We iterate between estimating posteriors over
    true labels and refining the confusion matrices, then impute missing worker labels.
    """

    def __init__(
        self,
        max_iters: int = 10,
        smoothing: float = 1e-2,
        convergence_tol: float = 1e-4,
        num_workers: int = 1,
        show_progress: bool = False,
        cgmatch_in_em: bool = False,
        cgmatch_momentum: float = 0.9,
    ) -> None:
        self.max_iters = max_iters
        self.smoothing = max(smoothing, 0.0)
        self.convergence_tol = max(convergence_tol, 0.0)
        self.num_workers = max(1, num_workers)
        self.show_progress = show_progress
        self.cgmatch_in_em = cgmatch_in_em
        self.cgmatch_momentum = cgmatch_momentum
        self.last_difficulty_stats: pd.DataFrame | None = None

    def complete(self, answers: pd.DataFrame) -> pd.DataFrame:
        self._validate_answers(answers)
        clean = answers[["object", "worker", "response"]].dropna().copy()
        if clean.empty:
            return clean

        objects = sorted(clean["object"].unique(), key=lambda value: str(value))
        workers = sorted(clean["worker"].unique(), key=lambda value: str(value))
        labels = sorted(clean["response"].unique(), key=lambda value: str(value))
        obj_index = {obj: idx for idx, obj in enumerate(objects)}
        worker_index = {worker: idx for idx, worker in enumerate(workers)}
        label_index = {label: idx for idx, label in enumerate(labels)}
        num_objects, num_workers, num_labels = len(objects), len(workers), len(labels)

        observations: List[List[Tuple[int, int]]] = [[] for _ in range(num_objects)]
        for row in clean.itertuples(index=False):
            obj_idx = obj_index[row.object]
            worker_idx = worker_index[row.worker]
            label_idx = label_index[row.response]
            observations[obj_idx].append((worker_idx, label_idx))

        posteriors = self._initialize_posteriors(observations, num_objects, num_labels)
        confusions = self._initialize_confusions(observations, posteriors, num_workers, num_labels)
        count_history = np.zeros((num_objects, num_labels), dtype=float)
        tau_e = 0.0
        tau_a = 0.0
        last_confidence = np.zeros(num_objects, dtype=float)
        last_gap = np.zeros(num_objects, dtype=float)
        last_variability = np.zeros(num_objects, dtype=float)
        last_difficulty: List[str] = ["ambiguous"] * num_objects

        for _ in self._progress(range(self.max_iters), desc="EM iterations"):
            new_posteriors = self._e_step(observations, confusions, num_objects, num_labels)
            change = np.abs(new_posteriors - posteriors).max()
            posteriors = new_posteriors
            easy_mask = np.ones(num_objects, dtype=bool)
            if self.cgmatch_in_em:
                preds = np.argmax(posteriors, axis=1)
                last_confidence = np.max(posteriors, axis=1)
                last_variability = np.var(posteriors, axis=1) if num_labels > 1 else np.zeros(num_objects)
                for obj_idx, pred in enumerate(preds):
                    count_history[obj_idx, pred] += 1.0
                top1 = count_history.max(axis=1)
                top2 = np.partition(count_history, -2, axis=1)[:, -2] if num_labels > 1 else np.zeros(num_objects)
                total_counts = count_history.sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    last_gap = np.where(total_counts > 0, (top1 - top2) / total_counts, 0.0)
                mu_conf = float(last_confidence.mean()) if num_objects else 0.0
                mu_gap = float(last_gap.mean()) if num_objects else 0.0
                momentum = self.cgmatch_momentum
                tau_e = momentum * tau_e + (1.0 - momentum) * mu_conf
                tau_a = momentum * tau_a + (1.0 - momentum) * mu_gap
                easy_mask = last_confidence >= tau_e
                ambiguous_mask = (~easy_mask) & (last_gap >= tau_a)
                last_difficulty = [
                    "easy" if easy_mask[i] else "ambiguous" if ambiguous_mask[i] else "hard" for i in range(num_objects)
                ]
                if not easy_mask.any():
                    easy_mask = np.ones(num_objects, dtype=bool)
            confusions = self._m_step(observations, posteriors, num_workers, num_labels, easy_mask=easy_mask)
            if self.convergence_tol and change < self.convergence_tol:
                break

        completed = self._impute_missing(clean, observations, posteriors, confusions, objects, workers, labels)
        if self.cgmatch_in_em:
            self.last_difficulty_stats = pd.DataFrame(
                {
                    "object": objects,
                    "confidence": last_confidence,
                    "count_gap": last_gap,
                    "variability": last_variability,
                    "difficulty": last_difficulty,
                    "tau_e": tau_e,
                    "tau_a": tau_a,
                }
            )
        else:
            self.last_difficulty_stats = None
        return completed

    def _initialize_posteriors(
        self, observations: Sequence[List[Tuple[int, int]]], num_objects: int, num_labels: int
    ) -> np.ndarray:
        posteriors = np.full((num_objects, num_labels), 1.0 / num_labels, dtype=float)
        for obj_idx, obs in enumerate(observations):
            if not obs:
                continue
            label_counts = Counter(label_idx for _, label_idx in obs)
            max_count = max(label_counts.values())
            tied = [label for label, count in label_counts.items() if count == max_count]
            majority_label = min(tied)
            posteriors[obj_idx] = 0.0
            posteriors[obj_idx, majority_label] = 1.0
        return posteriors

    def _initialize_confusions(
        self,
        observations: Sequence[List[Tuple[int, int]]],
        posteriors: np.ndarray,
        num_workers: int,
        num_labels: int,
    ) -> np.ndarray:
        confusions = np.full((num_workers, num_labels, num_labels), self.smoothing, dtype=float)
        for obj_idx, obs in enumerate(observations):
            true_idx = int(np.argmax(posteriors[obj_idx]))
            for worker_idx, label_idx in obs:
                confusions[worker_idx, true_idx, label_idx] += 1.0
        return self._normalize_confusions(confusions)

    def _e_step(
        self,
        observations: Sequence[List[Tuple[int, int]]],
        confusions: np.ndarray,
        num_objects: int,
        num_labels: int,
    ) -> np.ndarray:
        min_prob = 1e-12
        posteriors = np.zeros((num_objects, num_labels), dtype=float)
        for obj_idx, obs in enumerate(self._progress(range(num_objects), desc="E-step")):
            obs = observations[obj_idx]
            if not obs:
                posteriors[obj_idx] = 1.0 / num_labels
                continue
            log_probs = np.zeros(num_labels, dtype=float)
            for true_idx in range(num_labels):
                total_log = 0.0
                for worker_idx, observed_idx in obs:
                    prob = confusions[worker_idx, true_idx, observed_idx]
                    total_log += np.log(max(prob, min_prob))
                log_probs[true_idx] = total_log
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            total = probs.sum()
            posteriors[obj_idx] = probs / total if total > 0 else 1.0 / num_labels
        return posteriors

    def _m_step(
        self,
        observations: Sequence[List[Tuple[int, int]]],
        posteriors: np.ndarray,
        num_workers: int,
        num_labels: int,
        easy_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        confusions = np.full((num_workers, num_labels, num_labels), self.smoothing, dtype=float)
        use_mask = easy_mask is not None
        for obj_idx, obs in enumerate(self._progress(range(len(observations)), desc="M-step")):
            if use_mask and not bool(easy_mask[obj_idx]):
                continue
            obs = observations[obj_idx]
            posterior = posteriors[obj_idx]
            for worker_idx, observed_idx in obs:
                for true_idx in range(num_labels):
                    confusions[worker_idx, true_idx, observed_idx] += posterior[true_idx]
        return self._normalize_confusions(confusions)

    @staticmethod
    def _normalize_confusions(confusions: np.ndarray) -> np.ndarray:
        normalized = confusions.copy()
        num_workers, num_labels, _ = normalized.shape
        for worker_idx in range(num_workers):
            for true_idx in range(num_labels):
                row = normalized[worker_idx, true_idx]
                total = row.sum()
                if total > 0:
                    normalized[worker_idx, true_idx] = row / total
                else:
                    normalized[worker_idx, true_idx] = 1.0 / num_labels
        return normalized

    def _impute_missing(
        self,
        observed_df: pd.DataFrame,
        observations: Sequence[List[Tuple[int, int]]],
        posteriors: np.ndarray,
        confusions: np.ndarray,
        objects: Sequence[Hashable],
        workers: Sequence[Hashable],
        labels: Sequence[Hashable],
    ) -> pd.DataFrame:
        existing = {(row.object, row.worker) for row in observed_df.itertuples(index=False)}
        completed_records: List[Tuple[Hashable, Hashable, Hashable]] = [
            (row.object, row.worker, row.response) for row in observed_df.itertuples(index=False)
        ]

        num_labels = len(labels)

        def _process_object(obj_idx: int) -> List[Tuple[Hashable, Hashable, Hashable]]:
            obj = objects[obj_idx]
            posterior = posteriors[obj_idx]
            new_rows: List[Tuple[Hashable, Hashable, Hashable]] = []
            for worker_idx, worker in enumerate(workers):
                if (obj, worker) in existing:
                    continue
                scores = posterior @ confusions[worker_idx]
                best_label_idx = int(np.argmax(scores)) if num_labels else 0
                new_rows.append((obj, worker, labels[best_label_idx]))
            return new_rows

        obj_iterable = self._progress(range(len(objects)), desc="Imputing")
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for rows in executor.map(_process_object, obj_iterable):
                    completed_records.extend(rows)
        else:
            for obj_idx in obj_iterable:
                completed_records.extend(_process_object(obj_idx))

        return self._records_to_dataframe(completed_records)

    def _records_to_dataframe(self, records: List[Tuple[Hashable, Hashable, Hashable]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["object", "worker", "response"])
        # Small datasets: construct directly.
        if not self.show_progress or tqdm is None or len(records) < 200_000:
            return pd.DataFrame(records, columns=["object", "worker", "response"])

        frames: List[pd.DataFrame] = []
        chunk_size = 500_000
        progress = tqdm(total=len(records), desc="Building DataFrame", leave=False)
        for start in range(0, len(records), chunk_size):
            end = min(start + chunk_size, len(records))
            frames.append(pd.DataFrame(records[start:end], columns=["object", "worker", "response"]))
            progress.update(end - start)
        progress.close()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    def _progress(self, iterable: Iterable, desc: str):
        if not self.show_progress or tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, leave=False)


class TiReMGELabelCompletion(BaseCompleter):
    """
    Graph-based completion that wraps the TiReMGE GCN model from run_tiremge.py.

    The model jointly embeds objects and workers on a bipartite graph constructed
    from the observed annotations, then predicts the most likely label per object.
    Missing worker responses are imputed with these predictions.
    """

    def __init__(
        self,
        max_steps: int = 200,
        learning_rate: float = 1e-2,
        show_progress: bool = False,
        seed: int | None = None,
        cgmatch_in_completion: bool = False,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.learning_rate = learning_rate
        self.show_progress = show_progress
        self.seed = seed
        self.cgmatch_in_completion = cgmatch_in_completion
        self.last_predictions: pd.Series | None = None
        self.last_difficulty_stats: pd.DataFrame | None = None

    def complete(self, answers: pd.DataFrame) -> pd.DataFrame:
        self._validate_answers(answers)
        clean = answers[["object", "worker", "response"]].dropna().copy()
        if clean.empty:
            self.last_predictions = pd.Series(dtype=object)
            self.last_difficulty_stats = None
            return clean

        clean["object"] = clean["object"].astype(str)
        clean["worker"] = clean["worker"].astype(str)
        clean["response"] = clean["response"].astype(str)

        (
            objects,
            workers,
            labels,
            predictions,
            probabilities,
        ) = run_tiremge_prediction(
            clean,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            show_progress=self.show_progress,
            seed=self.seed,
            desc="TiReMGE completion",
        )

        completed_records = [
            (row.object, row.worker, row.response) for row in clean.itertuples(index=False)
        ]
        existing = {(row.object, row.worker) for row in clean.itertuples(index=False)}
        for obj in objects:
            pred = predictions[obj]
            for worker in workers:
                if (obj, worker) in existing:
                    continue
                completed_records.append((obj, worker, pred))

        completed_df = pd.DataFrame(completed_records, columns=["object", "worker", "response"])
        self.last_predictions = pd.Series([predictions[obj] for obj in objects], index=objects)
        if self.cgmatch_in_completion:
            self.last_difficulty_stats = build_difficulty_stats(objects, probabilities)
        else:
            self.last_difficulty_stats = None
        return completed_df
