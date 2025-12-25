from __future__ import annotations

import numpy as np
import pandas as pd

from truth_inference.completers_base_new import BaseCompleter


def _load_gcn_module():
    from truth_inference import gcn_new

    return gcn_new


class GCNLabelCompletion(BaseCompleter):
    """CGMatch-aware completion using a PyG GCN encoder and pairwise decoder."""

    def __init__(
        self,
        max_steps: int = 200,
        learning_rate: float = 1e-2,
        show_progress: bool = False,
        seed: int | None = None,
        cgmatch_in_completion: bool = False,
        cgmatch_momentum: float = 0.9,
        cgmatch_warmup_steps: int = 0,
        ambiguous_weight: float = 0.5,
        hard_weight: float = 0.0,
        reliability_alpha: float = 1.0,
        reliability_ema: float = 0.9,
        reliability_smoothing: float = 1.0,
        reliability_temperature: float = 1.0,
        object_loss_weight: float = 1.0,
        edge_loss_weight: float = 1.0,
        warmup_edge_weight: float = 0.0,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        device: str | None = None,
        hard_completion: str = "skip",
        completion_easy_confidence: float = 0.8,
        completion_gap_threshold: float = 0.15,
        use_cgmatch_thresholds: bool = True,
        completion_auto_thresholds: bool = False,
        completion_easy_quantile: float = 0.75,
        completion_gap_quantile: float = 0.75,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.learning_rate = learning_rate
        self.show_progress = show_progress
        self.seed = seed
        self.cgmatch_in_completion = cgmatch_in_completion
        self.cgmatch_momentum = cgmatch_momentum
        self.cgmatch_warmup_steps = max(0, int(cgmatch_warmup_steps))
        self.ambiguous_weight = ambiguous_weight
        self.hard_weight = hard_weight
        self.reliability_alpha = reliability_alpha
        self.reliability_ema = reliability_ema
        self.reliability_smoothing = reliability_smoothing
        self.reliability_temperature = reliability_temperature
        self.object_loss_weight = object_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.warmup_edge_weight = warmup_edge_weight
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.hard_completion = hard_completion
        self.completion_easy_confidence = completion_easy_confidence
        self.completion_gap_threshold = completion_gap_threshold
        self.use_cgmatch_thresholds = use_cgmatch_thresholds
        self.completion_auto_thresholds = completion_auto_thresholds
        self.completion_easy_quantile = completion_easy_quantile
        self.completion_gap_quantile = completion_gap_quantile
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

        gcn_new = _load_gcn_module()
        (
            objects,
            workers,
            labels,
            object_probs,
            pairwise_pred_indices,
            tau_e,
            tau_a,
        ) = gcn_new.run_gcn_completion(
            clean,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            show_progress=self.show_progress,
            seed=self.seed,
            cgmatch_in_completion=self.cgmatch_in_completion,
            cgmatch_momentum=self.cgmatch_momentum,
            cgmatch_warmup_steps=self.cgmatch_warmup_steps,
            ambiguous_weight=self.ambiguous_weight,
            hard_weight=self.hard_weight,
            reliability_alpha=self.reliability_alpha,
            reliability_ema=self.reliability_ema,
            reliability_smoothing=self.reliability_smoothing,
            reliability_temperature=self.reliability_temperature,
            object_loss_weight=self.object_loss_weight,
            edge_loss_weight=self.edge_loss_weight,
            warmup_edge_weight=self.warmup_edge_weight,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            device=self.device,
            desc="GCN completion",
        )

        object_pred_indices = object_probs.argmax(axis=1)
        confidence = object_probs.max(axis=1)
        if object_probs.shape[1] > 1:
            sorted_probs = np.sort(object_probs, axis=1)
            gap = sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            gap = np.zeros_like(confidence)
        if self.completion_auto_thresholds and len(confidence) > 0:
            easy_conf = float(np.quantile(confidence, self.completion_easy_quantile))
            gap_thresh = float(np.quantile(gap, self.completion_gap_quantile))
        elif self.cgmatch_in_completion and self.use_cgmatch_thresholds and (tau_e > 0.0 or tau_a > 0.0):
            easy_conf = float(tau_e)
            gap_thresh = float(tau_a)
        else:
            easy_conf = self.completion_easy_confidence
            gap_thresh = self.completion_gap_threshold
        difficulties = []
        for conf, delta in zip(confidence, gap):
            if conf >= easy_conf:
                difficulties.append("easy")
            elif delta >= gap_thresh:
                difficulties.append("ambiguous")
            else:
                difficulties.append("hard")

        completed_records = [
            (row.object, row.worker, row.response) for row in clean.itertuples(index=False)
        ]
        existing = {(row.object, row.worker) for row in clean.itertuples(index=False)}
        for obj_idx, obj in enumerate(objects):
            for worker_idx, worker in enumerate(workers):
                if (obj, worker) in existing:
                    continue
                if difficulties[obj_idx] == "hard" and self.hard_completion == "skip":
                    continue
                if difficulties[obj_idx] == "hard" and self.hard_completion == "object":
                    label_idx = int(object_pred_indices[obj_idx])
                else:
                    label_idx = int(pairwise_pred_indices[obj_idx, worker_idx])
                completed_records.append((obj, worker, labels[label_idx]))

        completed_df = pd.DataFrame(completed_records, columns=["object", "worker", "response"])
        self.last_predictions = pd.Series(
            [labels[int(idx)] for idx in object_pred_indices],
            index=objects,
        )
        if self.cgmatch_in_completion:
            self.last_difficulty_stats = gcn_new.build_difficulty_stats(objects, object_probs)
        else:
            self.last_difficulty_stats = None
        return completed_df
