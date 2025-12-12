from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm


class BaseJointModel(ABC):
    @abstractmethod
    def run(self, answers: pd.DataFrame) -> Dict[str, Any]:
        """Jointly infer true labels and complete missing annotations."""


class EMJointModel(BaseJointModel):
    """
    Dawid-Skene style joint EM:
    - latent true labels Y per object
    - worker confusion matrices theta_j[true, observed]
    - imputes missing worker-task labels using current posteriors
    """

    def __init__(
        self,
        max_iters: int = 10,
        smoothing: float = 1e-2,
        convergence_tol: float = 1e-4,
        num_workers: int = 1,
        show_progress: bool = False,
        missing_weight: float = 0.5,
    ) -> None:
        self.max_iters = max_iters
        self.smoothing = max(smoothing, 0.0)
        self.convergence_tol = max(convergence_tol, 0.0)
        self.num_workers = max(1, num_workers)
        self.show_progress = show_progress
        self.missing_weight = max(0.0, float(missing_weight))

    def run(self, answers: pd.DataFrame) -> Dict[str, Any]:
        self._validate_columns(answers)
        clean = answers[["object", "worker", "response"]].dropna().copy()
        objects = sorted(clean["object"].unique(), key=lambda value: str(value))
        workers = sorted(clean["worker"].unique(), key=lambda value: str(value))
        labels = sorted(clean["response"].unique(), key=self._label_sort_key)

        obj_index = {obj: idx for idx, obj in enumerate(objects)}
        worker_index = {worker: idx for idx, worker in enumerate(workers)}
        label_index = {label: idx for idx, label in enumerate(labels)}
        num_objects, num_workers, num_labels = len(objects), len(workers), len(labels)

        observations: List[List[Tuple[int, int]]] = [[] for _ in range(num_objects)]
        worker_obs: List[List[Tuple[int, int]]] = [[] for _ in range(num_workers)]
        existing_pairs = set()
        for row in clean.itertuples(index=False):
            oi = obj_index[row.object]
            wi = worker_index[row.worker]
            li = label_index[row.response]
            observations[oi].append((wi, li))
            worker_obs[wi].append((oi, li))
            existing_pairs.add((oi, wi))

        missing_by_worker = self._compute_missing_by_worker(worker_obs, num_objects)

        posteriors = self._initialize_posteriors(observations, num_objects, num_labels)
        confusions = self._initialize_confusions(observations, posteriors, num_workers, num_labels)

        prev_ll: float | None = None
        for _ in self._progress(range(self.max_iters), desc="Joint EM iterations"):
            posteriors = self._e_step(observations, confusions, num_objects, num_labels)
            confusions = self._m_step(
                worker_obs,
                missing_by_worker,
                posteriors,
                confusions,
                num_workers,
                num_labels,
            )
            curr_ll = self._log_likelihood(observations, confusions, num_labels)
            if prev_ll is not None and self.convergence_tol and abs(curr_ll - prev_ll) < self.convergence_tol:
                break
            prev_ll = curr_ll

        completed_records = self._impute_missing(
            observations,
            existing_pairs,
            posteriors,
            confusions,
            objects,
            workers,
            labels,
        )
        completed_answers = self._records_to_dataframe(completed_records)

        predictions = self._predict_labels(posteriors, labels, objects)
        return {
            "completed_answers": completed_answers,
            "predictions": predictions,
            "posteriors": posteriors,
            "confusions": confusions,
        }

    def _initialize_posteriors(
        self, observations: Sequence[List[Tuple[int, int]]], num_objects: int, num_labels: int
    ) -> np.ndarray:
        posteriors = np.full((num_objects, num_labels), 1.0 / num_labels, dtype=float)
        for obj_idx, obs in enumerate(observations):
            if not obs:
                continue
            counts = [0 for _ in range(num_labels)]
            for _, label_idx in obs:
                counts[label_idx] += 1
            max_count = max(counts)
            majority_label = min(idx for idx, val in enumerate(counts) if val == max_count)
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
            if not obs:
                continue
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
        for obj_idx in self._progress(range(num_objects), desc="E-step"):
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
        worker_obs: Sequence[List[Tuple[int, int]]],
        missing_by_worker: Sequence[np.ndarray],
        posteriors: np.ndarray,
        current_confusions: np.ndarray,
        num_workers: int,
        num_labels: int,
    ) -> np.ndarray:
        confusions = np.full((num_workers, num_labels, num_labels), self.smoothing, dtype=float)

        worker_iter = self._progress(range(num_workers), desc="M-step")
        for worker_idx in worker_iter:
            counts = confusions[worker_idx]

            # Observed contributions
            for obj_idx, observed_idx in worker_obs[worker_idx]:
                counts[:, observed_idx] += posteriors[obj_idx]

            # Missing contributions (soft expectation, optionally down-weighted)
            missing_indices = missing_by_worker[worker_idx]
            if missing_indices.size > 0 and self.missing_weight > 0:
                post_slice = posteriors[missing_indices]  # (m, L)
                q_missing = post_slice @ current_confusions[worker_idx]  # (m, L)
                counts += self.missing_weight * (post_slice.T @ q_missing)  # (L, L)

            # Normalize rows for this worker
            total_per_true = counts.sum(axis=1, keepdims=True)
            safe_total = np.where(total_per_true > 0, total_per_true, 1.0)
            confusions[worker_idx] = counts / safe_total

        return confusions

    def _log_likelihood(
        self,
        observations: Sequence[List[Tuple[int, int]]],
        confusions: np.ndarray,
        num_labels: int,
    ) -> float:
        """Observed-data log-likelihood with uniform prior over true labels."""
        if num_labels == 0:
            return float("-inf")
        prior = np.full(num_labels, 1.0 / num_labels, dtype=float)
        ll = 0.0
        min_prob = 1e-12
        for obs in observations:
            probs = prior.copy()
            for worker_idx, observed_idx in obs:
                probs *= confusions[worker_idx, :, observed_idx]
            ll += np.log(max(probs.sum(), min_prob))
        return float(ll)

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
        observations: Sequence[List[Tuple[int, int]]],
        existing_pairs: set,
        posteriors: np.ndarray,
        confusions: np.ndarray,
        objects: Sequence[Hashable],
        workers: Sequence[Hashable],
        labels: Sequence[Hashable],
    ) -> List[Tuple[Hashable, Hashable, Hashable]]:
        records: List[Tuple[Hashable, Hashable, Hashable]] = []
        for obj_idx, obs in enumerate(observations):
            obj = objects[obj_idx]
            for worker_idx, label_idx in obs:
                records.append((obj, workers[worker_idx], labels[label_idx]))

        num_labels = len(labels)

        def _process_object(obj_idx: int) -> List[Tuple[Hashable, Hashable, Hashable]]:
            obj = objects[obj_idx]
            posterior = posteriors[obj_idx]
            new_rows: List[Tuple[Hashable, Hashable, Hashable]] = []
            for worker_idx, worker in enumerate(workers):
                if (obj_idx, worker_idx) in existing_pairs:
                    continue
                scores = posterior @ confusions[worker_idx]
                best_label_idx = int(np.argmax(scores)) if num_labels else 0
                new_rows.append((obj, worker, labels[best_label_idx]))
            return new_rows

        iterable = self._progress(range(len(objects)), desc="Imputing missing")
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for rows in executor.map(_process_object, iterable):
                    records.extend(rows)
        else:
            for obj_idx in iterable:
                records.extend(_process_object(obj_idx))
        return records

    def _compute_missing_by_worker(self, worker_obs: Sequence[List[Tuple[int, int]]], num_objects: int) -> List[np.ndarray]:
        """Pre-compute missing object indices for each worker."""
        all_indices = np.arange(num_objects, dtype=int)
        missing: List[np.ndarray] = []
        for obs in worker_obs:
            mask = np.ones(num_objects, dtype=bool)
            for obj_idx, _ in obs:
                mask[obj_idx] = False
            missing.append(all_indices[mask])
        return missing

    def _records_to_dataframe(self, records: List[Tuple[Hashable, Hashable, Hashable]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["object", "worker", "response"])
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

    def _predict_labels(self, posteriors: np.ndarray, labels: Sequence[Hashable], objects: Sequence[Hashable]) -> pd.Series:
        predictions = {}
        for obj_idx, obj in enumerate(objects):
            posterior = posteriors[obj_idx]
            best_label_idx = int(np.argmax(posterior)) if len(labels) else 0
            predictions[obj] = labels[best_label_idx]
        return pd.Series(predictions)

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = {"object", "worker", "response"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in answers: {sorted(missing)}")

    def _progress(self, iterable: Iterable, desc: str):
        if not self.show_progress or tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, leave=False)

    @staticmethod
    def _label_sort_key(value: Hashable):
        """Sort labels numerically when possible, else lexicographically."""
        try:
            numeric = float(value)
            return (0, numeric)
        except (TypeError, ValueError):
            return (1, str(value))
