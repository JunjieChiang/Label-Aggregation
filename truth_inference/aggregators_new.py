from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Hashable, Iterable, Optional, Sequence, Tuple

import numpy as np

import pandas as pd

try:  # Optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional
    tqdm = None


class BaseAggregator(ABC):
    @abstractmethod
    def aggregate(self, completed_answers: pd.DataFrame) -> pd.Series:
        """
        Compute inferred true labels from a completed set of worker answers.

        Args:
            completed_answers: DataFrame with at least columns ['object', 'worker', 'response'].
        Returns:
            A Series indexed by task/object id with the inferred label for each task.
        """


def _build_ldplc_dataset(completed_answers: pd.DataFrame):
    from ldplc import AttributeMeta, Category, Dataset, Example, Label, Worker

    clean = completed_answers[["object", "worker", "response"]].dropna().copy()
    if clean.empty:
        empty_dataset = Dataset([AttributeMeta("example_index"), AttributeMeta("class", is_nominal=True, values=["0"])])
        empty_dataset.addCategory(Category("0", "0"))
        return empty_dataset, []

    objects = sorted(clean["object"].unique(), key=lambda value: str(value))
    workers = sorted(clean["worker"].unique(), key=lambda value: str(value))
    labels = sorted(clean["response"].unique(), key=lambda value: str(value))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    category_names = [str(idx) for idx in range(len(labels))]
    attribute_info = [
        AttributeMeta("example_index"),
        AttributeMeta("class", is_nominal=True, values=category_names),
    ]
    dataset = Dataset(attribute_info)
    for idx, name in enumerate(category_names):
        dataset.addCategory(Category(str(idx), name))

    object_index = {}
    for idx, object_id in enumerate(objects):
        example = Example(str(object_id), [float(idx), 0.0])
        dataset.addExample(example)
        object_index[object_id] = idx

    worker_map = {}
    for worker_id in workers:
        worker = Worker(str(worker_id))
        dataset.addWorker(worker)
        worker_map[worker_id] = worker

    for row in clean.itertuples(index=False):
        obj_idx = object_index[row.object]
        example = dataset.getExampleByIndex(obj_idx)
        worker = worker_map[row.worker]
        label_value = label_to_idx[row.response]
        label = Label(None, label_value, example.getId(), worker.getId())
        example.addNoisyLabel(label)
        worker.addNoisyLabel(label)

    return dataset, labels


def _majority_vote_predictions(dataset, labels: Sequence[str]) -> pd.Series:
    if not labels:
        return pd.Series(dtype=object)
    num_classes = len(labels)
    predictions = {}
    for example in dataset.examples:
        counts = [0 for _ in range(num_classes)]
        for label in example.noisy_labels.values():
            value = label.getValue()
            if 0 <= value < num_classes:
                counts[value] += 1
        if not any(counts):
            pred_idx = 0
        else:
            pred_idx = max(range(num_classes), key=lambda idx: counts[idx])
        predictions[example.getId()] = labels[pred_idx]
    return pd.Series(predictions)


class MajorityVoteAggregator(BaseAggregator):
    def __init__(self, num_workers: int = 1, show_progress: bool = False) -> None:
        self.num_workers = max(1, num_workers)
        self.show_progress = show_progress

    def aggregate(self, completed_answers: pd.DataFrame) -> pd.Series:
        self._validate_columns(completed_answers)
        predictions = {}
        object_ids = sorted(completed_answers["object"].unique(), key=lambda value: str(value))

        def _predict(object_id) -> Tuple[Hashable, Hashable]:
            labels = completed_answers.loc[completed_answers["object"] == object_id, "response"]
            return object_id, self._majority_label(labels)

        total = len(object_ids)
        if self.num_workers > 1:
            progress = self._progress_bar(total, desc="Aggregating")
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for object_id, pred in executor.map(_predict, object_ids):
                    predictions[object_id] = pred
                    if progress is not None:
                        progress.update()
            if progress is not None:
                progress.close()
        else:
            iterable = self._progress(object_ids, desc="Aggregating", total=total)
            for object_id in iterable:
                oid, pred = _predict(object_id)
                predictions[oid] = pred
        return pd.Series(predictions)

    @staticmethod
    def _majority_label(labels: Iterable[Hashable]) -> Hashable:
        counter = Counter(labels)
        if not counter:
            raise ValueError("No labels available to aggregate for a task.")
        max_votes = max(counter.values())
        tied = [label for label, count in counter.items() if count == max_votes]
        return sorted(tied, key=lambda value: str(value))[0]

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = {"object", "worker", "response"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in completed_answers: {sorted(missing)}")

    def _progress(self, iterable: Iterable, desc: str, total: int | None = None):
        if not self.show_progress or tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, total=total, leave=False)

    def _progress_bar(self, total: int, desc: str):
        if not self.show_progress or tqdm is None:
            return None
        return tqdm(total=total, desc=desc, leave=False)


class TiReMGEAggregator(BaseAggregator):
    def __init__(
        self,
        max_steps: int = 200,
        learning_rate: float = 1e-2,
        show_progress: bool = False,
        seed: int | None = None,
        cgmatch_stats: bool = False,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.learning_rate = learning_rate
        self.show_progress = show_progress
        self.seed = seed
        self.cgmatch_stats = cgmatch_stats
        self.last_difficulty_stats: pd.DataFrame | None = None

    def aggregate(self, completed_answers: pd.DataFrame) -> pd.Series:
        MajorityVoteAggregator._validate_columns(completed_answers)
        clean = completed_answers[["object", "worker", "response"]].dropna().copy()
        if clean.empty:
            self.last_difficulty_stats = None
            return pd.Series(dtype=object)
        clean["object"] = clean["object"].astype(str)
        clean["worker"] = clean["worker"].astype(str)
        clean["response"] = clean["response"].astype(str)

        from truth_inference.tiremge import build_difficulty_stats, run_tiremge_prediction

        objects, _, _, predictions, probabilities = run_tiremge_prediction(
            clean,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            show_progress=self.show_progress,
            seed=self.seed,
            desc="TiReMGE aggregation",
        )
        if self.cgmatch_stats:
            self.last_difficulty_stats = build_difficulty_stats(objects, probabilities)
        else:
            self.last_difficulty_stats = None
        return pd.Series([predictions[obj] for obj in objects], index=objects)


class DawidSkeneAggregator(BaseAggregator):
    def __init__(
        self,
        max_iters: int = 50,
        smoothing: float = 1e-2,
        convergence_tol: float = 1e-5,
        show_progress: bool = False,
    ) -> None:
        self.max_iters = max(1, int(max_iters))
        self.smoothing = max(smoothing, 0.0)
        self.convergence_tol = max(convergence_tol, 0.0)
        self.show_progress = show_progress

    def aggregate(self, completed_answers: pd.DataFrame) -> pd.Series:
        MajorityVoteAggregator._validate_columns(completed_answers)
        clean = completed_answers[["object", "worker", "response"]].dropna().copy()
        if clean.empty:
            return pd.Series(dtype=object)

        objects = sorted(clean["object"].unique(), key=lambda value: str(value))
        workers = sorted(clean["worker"].unique(), key=lambda value: str(value))
        labels = sorted(clean["response"].unique(), key=lambda value: str(value))
        obj_index = {obj: idx for idx, obj in enumerate(objects)}
        worker_index = {worker: idx for idx, worker in enumerate(workers)}
        label_index = {label: idx for idx, label in enumerate(labels)}
        num_objects, num_workers, num_labels = len(objects), len(workers), len(labels)

        observations = [[] for _ in range(num_objects)]
        for row in clean.itertuples(index=False):
            obj_idx = obj_index[row.object]
            worker_idx = worker_index[row.worker]
            label_idx = label_index[row.response]
            observations[obj_idx].append((worker_idx, label_idx))

        posteriors = self._initialize_posteriors(observations, num_objects, num_labels)
        confusions = self._initialize_confusions(observations, posteriors, num_workers, num_labels)

        for _ in self._progress(range(self.max_iters), desc="DS iterations"):
            new_posteriors = self._e_step(observations, confusions, num_objects, num_labels)
            change = np.abs(new_posteriors - posteriors).max()
            posteriors = new_posteriors
            confusions = self._m_step(observations, posteriors, num_workers, num_labels)
            if self.convergence_tol and change < self.convergence_tol:
                break

        pred_indices = np.argmax(posteriors, axis=1) if num_labels > 0 else np.zeros(num_objects, dtype=int)
        predictions = [labels[int(idx)] for idx in pred_indices]
        return pd.Series(predictions, index=objects)

    def _initialize_posteriors(self, observations, num_objects: int, num_labels: int) -> np.ndarray:
        posteriors = np.full((num_objects, num_labels), 1.0 / num_labels, dtype=float)
        for obj_idx, obs in enumerate(observations):
            if not obs:
                continue
            counts = Counter(label_idx for _, label_idx in obs)
            max_count = max(counts.values())
            tied = [label for label, count in counts.items() if count == max_count]
            majority_label = min(tied)
            posteriors[obj_idx] = 0.0
            posteriors[obj_idx, majority_label] = 1.0
        return posteriors

    def _initialize_confusions(
        self,
        observations,
        posteriors: np.ndarray,
        num_workers: int,
        num_labels: int,
    ) -> np.ndarray:
        confusions = np.full((num_workers, num_labels, num_labels), self.smoothing, dtype=float)
        for obj_idx, obs in enumerate(observations):
            true_idx = int(np.argmax(posteriors[obj_idx])) if num_labels > 0 else 0
            for worker_idx, label_idx in obs:
                confusions[worker_idx, true_idx, label_idx] += 1.0
        return self._normalize_confusions(confusions)

    def _e_step(
        self,
        observations,
        confusions: np.ndarray,
        num_objects: int,
        num_labels: int,
    ) -> np.ndarray:
        min_prob = 1e-12
        posteriors = np.zeros((num_objects, num_labels), dtype=float)
        for obj_idx in self._progress(range(num_objects), desc="DS E-step"):
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
        observations,
        posteriors: np.ndarray,
        num_workers: int,
        num_labels: int,
    ) -> np.ndarray:
        confusions = np.full((num_workers, num_labels, num_labels), self.smoothing, dtype=float)
        for obj_idx, obs in enumerate(self._progress(range(len(observations)), desc="DS M-step")):
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

    def _progress(self, iterable: Iterable, desc: str):
        if not self.show_progress or tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, leave=False)


class WSLCAggregator(BaseAggregator):
    def __init__(
        self,
        num_neighbors: Optional[int] = None,
        embedding_mode: str = "manual",
        graphsage_dim: Optional[int] = None,
        graphsage_layers: int = 2,
    ) -> None:
        self.num_neighbors = num_neighbors
        self.embedding_mode = embedding_mode
        self.graphsage_dim = graphsage_dim
        self.graphsage_layers = max(1, int(graphsage_layers))

    def aggregate(self, completed_answers: pd.DataFrame) -> pd.Series:
        MajorityVoteAggregator._validate_columns(completed_answers)
        dataset, labels = _build_ldplc_dataset(completed_answers)
        if dataset.getExampleSize() == 0:
            return pd.Series(dtype=object)
        from wslc import WSLC

        WSLC(
            num_neighbors=self.num_neighbors,
            embedding_mode=self.embedding_mode,
            graphsage_dim=self.graphsage_dim,
            graphsage_layers=self.graphsage_layers,
        ).do_inference(dataset)
        return _majority_vote_predictions(dataset, labels)


class LDPLCAggregator(BaseAggregator):
    def __init__(self, k: int = 5, iterations: int = 5, num_threads: int = 1) -> None:
        self.k = max(0, int(k))
        self.iterations = max(1, int(iterations))
        self.num_threads = max(1, int(num_threads))

    def aggregate(self, completed_answers: pd.DataFrame) -> pd.Series:
        MajorityVoteAggregator._validate_columns(completed_answers)
        dataset, labels = _build_ldplc_dataset(completed_answers)
        if dataset.getExampleSize() == 0:
            return pd.Series(dtype=object)
        from ldplc import LDPLC

        LDPLC(k=self.k, iterations=self.iterations, num_threads=self.num_threads).doInference(dataset)
        return _majority_vote_predictions(dataset, labels)
