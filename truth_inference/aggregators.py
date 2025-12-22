from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Hashable, Iterable, Mapping, Sequence, Tuple
import pandas as pd

from truth_inference.tiremge import build_difficulty_stats, run_tiremge_prediction

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
        # Tie-break deterministically using lexicographic order on string form.
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
