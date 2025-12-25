from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv
except ImportError as exc:  # pragma: no cover - optional
    raise ImportError("GCN completion requires PyTorch and PyTorch Geometric.") from exc

try:  # Optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional
    tqdm = None


def _encode_entities(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    objects = sorted(df["object"].unique(), key=lambda value: str(value))
    workers = sorted(df["worker"].unique(), key=lambda value: str(value))
    labels = sorted(df["response"].unique(), key=lambda value: str(value))
    obj_to_idx = {obj: idx for idx, obj in enumerate(objects)}
    worker_to_idx = {worker: idx for idx, worker in enumerate(workers)}
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    object_indices = df["object"].map(obj_to_idx).to_numpy(dtype=np.int64)
    worker_indices = df["worker"].map(worker_to_idx).to_numpy(dtype=np.int64)
    label_indices = df["response"].map(label_to_idx).to_numpy(dtype=np.int64)
    return objects, workers, labels, object_indices, worker_indices, label_indices


def _progress_iter(num_steps: int, show_progress: bool, desc: str):
    iterable = range(num_steps)
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def _row_normalize(features: torch.Tensor) -> torch.Tensor:
    denom = features.sum(dim=1, keepdim=True)
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return features / denom


def _build_features(
    object_indices: torch.Tensor,
    source_indices: torch.Tensor,
    label_indices: torch.Tensor,
    reliability: torch.Tensor,
    num_objects: int,
    num_sources: int,
    num_classes: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    one_hot = F.one_hot(label_indices, num_classes=num_classes).float()
    weighted_claims = one_hot * reliability[source_indices].unsqueeze(1)

    object_features = torch.zeros((num_objects, num_classes), device=one_hot.device)
    object_features.index_add_(0, object_indices, weighted_claims)

    source_counts = torch.zeros((num_sources, num_classes), device=one_hot.device)
    source_counts.index_add_(0, source_indices, one_hot)
    source_fingerprint = _row_normalize((source_counts > 0).float()) * reliability.unsqueeze(1)

    return (
        {
            "object": _row_normalize(object_features),
            "source": source_fingerprint,
        },
        object_features,
    )


def _build_object_targets(object_features: torch.Tensor) -> torch.Tensor:
    totals = object_features.sum(dim=1, keepdim=True)
    num_classes = object_features.size(1)
    uniform = torch.full_like(object_features, 1.0 / float(num_classes))
    normalized = torch.where(totals > 0, object_features / totals, uniform)
    return normalized


def _confidence_gap(probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    confidence = probs.max(dim=1).values
    if probs.size(1) > 1:
        top2 = torch.topk(probs, k=2, dim=1).values
        gap = top2[:, 0] - top2[:, 1]
    else:
        gap = torch.zeros_like(confidence)
    return confidence, gap


def _difficulty_weights(
    confidence: torch.Tensor,
    gap: torch.Tensor,
    tau_e: float,
    tau_a: float,
    easy_weight: float,
    ambiguous_weight: float,
    hard_weight: float,
) -> torch.Tensor:
    easy_mask = confidence >= tau_e
    ambiguous_mask = (~easy_mask) & (gap >= tau_a)
    weights = torch.full_like(confidence, float(hard_weight))
    weights = torch.where(ambiguous_mask, torch.as_tensor(ambiguous_weight, device=weights.device), weights)
    weights = torch.where(easy_mask, torch.as_tensor(easy_weight, device=weights.device), weights)
    return weights


def _update_reliability(
    edge_loss: torch.Tensor,
    edge_obj_weights: torch.Tensor,
    source_indices: torch.Tensor,
    num_sources: int,
    alpha: float,
    ema: float,
    reliability: torch.Tensor,
) -> torch.Tensor:
    device = edge_loss.device
    sums = torch.zeros(num_sources, device=device)
    counts = torch.zeros(num_sources, device=device)
    masked_loss = edge_loss * edge_obj_weights
    sums.index_add_(0, source_indices, masked_loss)
    counts.index_add_(0, source_indices, (edge_obj_weights > 0).float())
    mean_loss = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
    r_new = torch.exp(-alpha * mean_loss)
    return ema * reliability + (1.0 - ema) * r_new


def _smooth_reliability(
    source_embeddings: torch.Tensor,
    reliability: torch.Tensor,
    temperature: float,
    smoothing: float,
) -> torch.Tensor:
    if smoothing <= 0.0:
        return reliability
    embeddings = F.normalize(source_embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.t()
    sim = sim / max(temperature, 1e-6)
    weights = F.softmax(sim, dim=1)
    smoothed = weights @ reliability
    return (1.0 - smoothing) * reliability + smoothing * smoothed


class HeteroGCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.branch_a = HeteroConv(
            {
                ("source", "claims", "object"): SAGEConv((-1, -1), hidden_dim),
                ("object", "rev_claims", "source"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="mean",
        )
        self.branch_b1 = HeteroConv(
            {
                ("source", "claims", "object"): SAGEConv((-1, -1), hidden_dim, root_weight=False),
                ("object", "rev_claims", "source"): SAGEConv((-1, -1), hidden_dim, root_weight=False),
            },
            aggr="mean",
        )
        self.branch_b2 = HeteroConv(
            {
                ("source", "claims", "object"): SAGEConv((-1, -1), hidden_dim, root_weight=False),
                ("object", "rev_claims", "source"): SAGEConv((-1, -1), hidden_dim, root_weight=False),
            },
            aggr="mean",
        )
        self.dropout = dropout

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        h_a = self.branch_a(x_dict, edge_index_dict)
        h_a = {key: F.relu(value) for key, value in h_a.items()}

        h_b = self.branch_b1(x_dict, edge_index_dict)
        h_b = {key: F.relu(value) for key, value in h_b.items()}
        h_b = self.branch_b2(h_b, edge_index_dict)
        h_b = {key: F.relu(value) for key, value in h_b.items()}

        merged = {key: h_a[key] + h_b[key] for key in h_a}
        return {key: F.dropout(value, p=self.dropout, training=self.training) for key, value in merged.items()}


class GCNCompletionModel(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = HeteroGCNEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.object_head = nn.Linear(hidden_dim, num_classes)
        self.pairwise_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.encoder(x_dict, edge_index_dict)

    def object_logits(self, object_embeddings: torch.Tensor) -> torch.Tensor:
        return self.object_head(object_embeddings)

    def pairwise_logits(
        self,
        source_embeddings: torch.Tensor,
        object_embeddings: torch.Tensor,
        source_indices: torch.Tensor,
        object_indices: torch.Tensor,
    ) -> torch.Tensor:
        hs = source_embeddings[source_indices]
        ho = object_embeddings[object_indices]
        features = torch.cat([hs, ho, hs * ho], dim=1)
        return self.pairwise_head(features)


def run_gcn_completion(
    answers: pd.DataFrame,
    *,
    max_steps: int,
    learning_rate: float,
    show_progress: bool,
    seed: int | None,
    cgmatch_in_completion: bool,
    cgmatch_momentum: float,
    cgmatch_warmup_steps: int,
    ambiguous_weight: float,
    hard_weight: float,
    reliability_alpha: float,
    reliability_ema: float,
    reliability_smoothing: float,
    reliability_temperature: float,
    object_loss_weight: float,
    edge_loss_weight: float,
    warmup_edge_weight: float,
    hidden_dim: int,
    dropout: float,
    device: str | None,
    desc: str = "GCN completion",
) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, float, float]:
    (
        objects,
        workers,
        labels,
        object_indices_np,
        worker_indices_np,
        label_indices_np,
    ) = _encode_entities(answers)

    object_num = len(objects)
    source_num = len(workers)
    class_num = len(labels)
    if object_num == 0 or source_num == 0:
        raise ValueError("GCN completion requires at least one object and one worker annotation.")
    if class_num == 0:
        raise ValueError("GCN completion requires at least one unique label.")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    resolved_device = device
    if resolved_device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(resolved_device)

    object_indices = torch.tensor(object_indices_np, dtype=torch.long, device=device_obj)
    source_indices = torch.tensor(worker_indices_np, dtype=torch.long, device=device_obj)
    label_indices = torch.tensor(label_indices_np, dtype=torch.long, device=device_obj)
    edge_index = torch.stack([source_indices, object_indices], dim=0)
    edge_index_dict = {
        ("source", "claims", "object"): edge_index,
        ("object", "rev_claims", "source"): edge_index.flip(0),
    }

    model = GCNCompletionModel(hidden_dim=hidden_dim, num_classes=class_num, dropout=dropout).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reliability = torch.full((source_num,), 1.0 / float(source_num), device=device_obj)
    tau_e = 0.0
    tau_a = 0.0

    for step in _progress_iter(max_steps, show_progress, desc):
        model.train()
        x_dict, object_feature_counts = _build_features(
            object_indices,
            source_indices,
            label_indices,
            reliability,
            object_num,
            source_num,
            class_num,
        )
        embeddings = model.encode(x_dict, edge_index_dict)
        object_embeddings = embeddings["object"]
        source_embeddings = embeddings["source"]
        object_logits = model.object_logits(object_embeddings)
        object_probs = F.softmax(object_logits, dim=1)
        conf, gap = _confidence_gap(object_probs.detach())
        object_targets = _build_object_targets(object_feature_counts.detach())
        log_probs = F.log_softmax(object_logits, dim=1)
        object_loss = -(object_targets * log_probs).sum(dim=1).mean()

        if cgmatch_in_completion and step >= cgmatch_warmup_steps:
            mean_conf = float(conf.mean().item())
            mean_gap = float(gap.mean().item())
            tau_e = cgmatch_momentum * tau_e + (1.0 - cgmatch_momentum) * mean_conf
            tau_a = cgmatch_momentum * tau_a + (1.0 - cgmatch_momentum) * mean_gap
            object_weights = _difficulty_weights(
                conf,
                gap,
                tau_e,
                tau_a,
                easy_weight=1.0,
                ambiguous_weight=ambiguous_weight,
                hard_weight=hard_weight,
            )
        else:
            object_weights = torch.ones_like(conf)

        edge_logits = model.pairwise_logits(
            source_embeddings,
            object_embeddings,
            source_indices,
            object_indices,
        )
        edge_loss = F.cross_entropy(edge_logits, label_indices, reduction="none")
        edge_obj_weight = object_weights[object_indices]
        edge_weight = edge_obj_weight * reliability[source_indices]
        edge_loss_value = (edge_loss * edge_weight).sum() / (edge_weight.sum() + 1e-8)
        current_edge_weight = edge_loss_weight if step >= cgmatch_warmup_steps else warmup_edge_weight
        loss = object_loss_weight * object_loss + current_edge_weight * edge_loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cgmatch_in_completion and step >= cgmatch_warmup_steps:
            reliability = _update_reliability(
                edge_loss.detach(),
                edge_obj_weight.detach(),
                source_indices,
                source_num,
                reliability_alpha,
                reliability_ema,
                reliability,
            )
            reliability = _smooth_reliability(
                source_embeddings.detach(),
                reliability,
                temperature=reliability_temperature,
                smoothing=reliability_smoothing,
            )
        elif not cgmatch_in_completion:
            reliability = _update_reliability(
                edge_loss.detach(),
                torch.ones_like(edge_obj_weight),
                source_indices,
                source_num,
                reliability_alpha,
                reliability_ema,
                reliability,
            )
            reliability = _smooth_reliability(
                source_embeddings.detach(),
                reliability,
                temperature=reliability_temperature,
                smoothing=reliability_smoothing,
            )

    model.eval()
    with torch.no_grad():
        x_dict, _ = _build_features(
            object_indices,
            source_indices,
            label_indices,
            reliability,
            object_num,
            source_num,
            class_num,
        )
        embeddings = model.encode(x_dict, edge_index_dict)
        object_embeddings = embeddings["object"]
        source_embeddings = embeddings["source"]
        object_logits = model.object_logits(object_embeddings)
        object_probs = F.softmax(object_logits, dim=1).cpu().numpy()

        pairwise_pred_indices = np.zeros((object_num, source_num), dtype=np.int64)
        for obj_idx in range(object_num):
            ho = object_embeddings[obj_idx].unsqueeze(0).repeat(source_num, 1)
            hs = source_embeddings
            features = torch.cat([hs, ho, hs * ho], dim=1)
            logits = model.pairwise_head(features)
            pairwise_pred_indices[obj_idx] = logits.argmax(dim=1).cpu().numpy()

    return objects, workers, labels, object_probs, pairwise_pred_indices, float(tau_e), float(tau_a)


def build_difficulty_stats(objects: Sequence[str], probs: np.ndarray) -> pd.DataFrame:
    if probs.size == 0:
        return pd.DataFrame(columns=["object", "confidence", "count_gap", "variability"])
    top1 = probs.max(axis=1)
    if probs.shape[1] > 1:
        sorted_probs = np.sort(probs, axis=1)
        top2 = sorted_probs[:, -2]
    else:
        top2 = np.zeros_like(top1)
    gap = top1 - top2
    variability = probs.var(axis=1)
    return pd.DataFrame(
        {
            "object": list(objects),
            "confidence": top1,
            "count_gap": gap,
            "variability": variability,
        }
    )
