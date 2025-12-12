from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ldplc import (
    AttributeMeta,
    Dataset,
    Example,
    Label,
    load_csv_dataset,
    load_labelme_dataset,
    load_trec_dataset,
    mean,
    normalize,
    _aggregator_display,
    _evaluate_dataset,
    iter_with_progress,
)


def _mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _calc_attribute_r(x: Sequence[float], y: Sequence[float]) -> float:
    mean_x = _mean(x)
    mean_y = _mean(y)
    numerator = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for xi, yi in zip(x, y):
        numerator += (xi - mean_x) * (yi - mean_y)
        denom_x += (xi - mean_x) ** 2
        denom_y += (yi - mean_y) ** 2
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / ((denom_x * denom_y) ** 0.5)


class WSLC:
    """Python port of the CEKA WSLC label completion algorithm."""

    def __init__(
        self,
        num_neighbors: Optional[int] = None,
        embedding_mode: str = "manual",
        graphsage_dim: Optional[int] = None,
        graphsage_layers: int = 2,
    ) -> None:
        self.num_neighbors = num_neighbors
        self.num_classes = 0
        self.num_workers = 0
        self.worker_ids: List[str] = []
        self.embedding_mode = embedding_mode
        self.graphsage_dim = graphsage_dim
        self.graphsage_layers = max(1, graphsage_layers)

    def do_inference(self, dataset: Dataset) -> Dataset:
        self.num_classes = dataset.numClasses()
        self.num_workers = dataset.getWorkerSize()
        self.worker_ids = [dataset.getWorkerByIndex(i).getId() for i in range(self.num_workers)]

        worker_examples: Dict[str, List[Example]] = {worker_id: [] for worker_id in self.worker_ids}
        for example in dataset.examples:
            for worker_id in example.noisy_labels.keys():
                worker_examples.setdefault(worker_id, []).append(example)

        if self.embedding_mode == "graphsage":
            attribute_vectors = self._calc_graphsage_worker_embeddings(
                dataset, worker_examples, dataset.attribute_info
            )
        else:
            attribute_vectors = {}
            for idx in iter_with_progress(
                range(self.num_workers), total=self.num_workers, desc="Computing worker features"
            ):
                worker_id = self.worker_ids[idx]
                attribute_vectors[worker_id] = self._calc_attribute_vector(
                    worker_examples.get(worker_id, []), worker_id, dataset.attribute_info
                )

        num_k = self.num_neighbors if self.num_neighbors and self.num_neighbors > 0 else max(self.num_workers - 1, 1)
        neighbour_indices: List[List[int]] = [[0 for _ in range(num_k)] for _ in range(self.num_workers)]
        neighbour_weights: List[List[float]] = [[0.0 for _ in range(num_k)] for _ in range(self.num_workers)]

        for i in iter_with_progress(
            range(self.num_workers), total=self.num_workers, desc="Computing worker similarity"
        ):
            similarities = []
            for j in range(self.num_workers):
                if i == j:
                    continue
                sim = self._calc_similarity(attribute_vectors[self.worker_ids[i]], attribute_vectors[self.worker_ids[j]])
                similarities.append((sim, j))
            similarities.sort(reverse=True, key=lambda pair: pair[0])
            top = similarities[:num_k]
            neighbour_indices[i] = [idx for _, idx in top]
            neighbour_weights[i] = [weight for weight, _ in top]

        num_examples = dataset.getExampleSize()
        for example_idx in iter_with_progress(
            range(num_examples), total=num_examples, desc="Completing labels"
        ):
            example = dataset.getExampleByIndex(example_idx)
            existing = set(example.noisy_labels.keys())
            for worker_idx, worker_id in enumerate(self.worker_ids):
                if worker_id in existing:
                    continue
                predicted = self._predict_label(neighbour_indices[worker_idx], neighbour_weights[worker_idx], example)
                label = Label(None, predicted, example.getId(), worker_id)
                example.addNoisyLabel(label)
                dataset.getWorkerByIndex(worker_idx).addNoisyLabel(label)
        return dataset

    def _predict_label(self, neighbour_indices: Sequence[int], neighbour_weights: Sequence[float], example: Example) -> int:
        votes = [0.0 for _ in range(self.num_classes)]
        for idx, worker_index in enumerate(neighbour_indices):
            worker_id = self.worker_ids[worker_index]
            label = example.getNoisyLabelByWorkerId(worker_id)
            if label is None:
                continue
            votes[label.getValue()] += neighbour_weights[idx]
        return votes.index(max(votes)) if any(votes) else 0

    def _calc_attribute_vector(self, examples: Sequence[Example], worker_id: str, attribute_info: List[AttributeMeta]) -> List[float]:
        num_examples = len(examples)
        result = [0.0 for _ in range(len(attribute_info) - 1)]
        if num_examples == 0:
            return result

        class_probs = [0.0 for _ in range(self.num_classes)]
        classy = [[0.0 for _ in range(num_examples)] for _ in range(self.num_classes)]
        for idx, example in enumerate(examples):
            label = example.getNoisyLabelByWorkerId(worker_id)
            class_idx = label.getValue() if label else 0
            class_probs[class_idx] += 1.0
            classy[class_idx][idx] = 1.0
        normalize(class_probs)

        for att_idx in range(len(attribute_info) - 1):
            meta = attribute_info[att_idx]
            if meta.is_nominal:
                num_att = len(meta.values) if meta.values else 0
                observed_values: List[int] = []
                for example in examples:
                    value = int(round(example.value(att_idx)))
                    observed_values.append(value)
                    num_att = max(num_att, value + 1)
                num_att = max(num_att, 1)
                attx = [[0.0 for _ in range(num_examples)] for _ in range(num_att)]
                paic = [[0.0 for _ in range(self.num_classes)] for _ in range(num_att)]
                for row_idx, value in enumerate(observed_values):
                    capped = max(0, min(value, num_att - 1))
                    label = examples[row_idx].getNoisyLabelByWorkerId(worker_id)
                    class_idx = label.getValue() if label else 0
                    paic[capped][class_idx] += 1.0
                    attx[capped][row_idx] = 1.0
                for class_idx in range(self.num_classes):
                    for value_idx in range(num_att):
                        paic[value_idx][class_idx] /= num_examples
                        if paic[value_idx][class_idx] != 0:
                            result[att_idx] += paic[value_idx][class_idx] * _calc_attribute_r(attx[value_idx], classy[class_idx])
            else:
                att_values = [example.value(att_idx) for example in examples]
                for class_idx in range(self.num_classes):
                    if class_probs[class_idx] != 0:
                        result[att_idx] += class_probs[class_idx] * _calc_attribute_r(att_values, classy[class_idx])
        return result

    def _calc_graphsage_worker_embeddings(
        self,
        dataset: Dataset,
        worker_examples: Dict[str, List[Example]],
        attribute_info: List[AttributeMeta],
    ) -> Dict[str, List[float]]:
        """Lightweight GraphSAGE-style mean aggregator over the worker-instance bipartite graph."""
        num_examples = dataset.getExampleSize()
        if num_examples == 0 or self.num_workers == 0:
            return {wid: [] for wid in self.worker_ids}

        feature_dim = max(len(attribute_info) - 1, 1)
        embed_dim = self.graphsage_dim if self.graphsage_dim and self.graphsage_dim > 0 else feature_dim
        total_nodes = self.num_workers + num_examples
        example_offset = self.num_workers

        # Base node features: workers start at zeros; examples use their attribute values.
        node_features: List[List[float]] = [[0.0 for _ in range(embed_dim)] for _ in range(total_nodes)]
        for idx, example in enumerate(dataset.examples):
            raw = [example.value(att_idx) for att_idx in range(feature_dim)]
            node_features[example_offset + idx] = self._pad_or_truncate(raw, embed_dim)

        # Build adjacency: worker <-> example edges where a worker labeled that example.
        adjacency: List[List[int]] = [[] for _ in range(total_nodes)]
        example_id_to_idx = {example.getId(): idx for idx, example in enumerate(dataset.examples)}
        for worker_idx, worker_id in enumerate(self.worker_ids):
            for example in worker_examples.get(worker_id, []):
                ex_idx = example_id_to_idx.get(example.getId())
                if ex_idx is None:
                    continue
                ex_node = example_offset + ex_idx
                adjacency[worker_idx].append(ex_node)
                adjacency[ex_node].append(worker_idx)

        # GraphSAGE mean aggregation with shared weights (identity) for a few layers.
        h = node_features
        for _ in range(self.graphsage_layers):
            new_h: List[List[float]] = []
            for node_idx in range(total_nodes):
                neigh_indices = adjacency[node_idx]
                if neigh_indices:
                    neigh_mean = [0.0 for _ in range(embed_dim)]
                    for neigh in neigh_indices:
                        for dim in range(embed_dim):
                            neigh_mean[dim] += h[neigh][dim]
                    factor = 1.0 / len(neigh_indices)
                    for dim in range(embed_dim):
                        neigh_mean[dim] *= factor
                else:
                    neigh_mean = [0.0 for _ in range(embed_dim)]

                combined = [h[node_idx][dim] + neigh_mean[dim] for dim in range(embed_dim)]
                # ReLU
                for dim in range(embed_dim):
                    if combined[dim] < 0:
                        combined[dim] = 0.0
                new_h.append(combined)
            h = new_h

        # Collect worker embeddings, L2-normalize for cosine stability.
        worker_vectors: Dict[str, List[float]] = {}
        for worker_idx, worker_id in enumerate(self.worker_ids):
            vec = h[worker_idx]
            norm = sum(v * v for v in vec) ** 0.5
            if norm != 0:
                vec = [v / norm for v in vec]
            worker_vectors[worker_id] = vec
        return worker_vectors

    @staticmethod
    def _pad_or_truncate(values: List[float], target: int) -> List[float]:
        if len(values) >= target:
            return list(values[:target])
        return list(values) + [0.0 for _ in range(target - len(values))]

    @staticmethod
    def _calc_similarity(vec_one: Sequence[float], vec_two: Sequence[float]) -> float:
        numerator = 0.0
        denom_one = 0.0
        denom_two = 0.0
        for a, b in zip(vec_one, vec_two):
            numerator += a * b
            denom_one += a * a
            denom_two += b * b
        if numerator == 0 or denom_one == 0 or denom_two == 0:
            return 0.0
        cosine = numerator / (denom_one ** 0.5 * denom_two ** 0.5)
        return 0.5 * (1.0 + cosine)


def _run_dataset(
    dataset: Dataset,
    num_neighbors: Optional[int],
    aggregator: str,
    aggregator_opts: Optional[Dict[str, float]],
    embedding_mode: str,
    graphsage_dim: Optional[int],
    graphsage_layers: int,
) -> None:
    label = _aggregator_display(aggregator)
    baseline = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(
        f"Baseline {label} accuracy (before WSLC): {baseline * 100:.2f}% "
        f"(examples={dataset.getExampleSize()}, workers={dataset.getWorkerSize()}, classes={dataset.numClasses()})"
    )
    algorithm = WSLC(
        num_neighbors=num_neighbors,
        embedding_mode=embedding_mode,
        graphsage_dim=graphsage_dim,
        graphsage_layers=graphsage_layers,
    )
    algorithm.do_inference(dataset)
    post = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Post-WSLC {label} accuracy: {post * 100:.2f}%")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WSLC label completion (Python port)")
    parser.add_argument("--labelme", action="store_true", help="Run WSLC on the LabelMe dataset")
    parser.add_argument("--labelme-root", default="datasets/real-world/labelme", help="Path to LabelMe data directory")
    parser.add_argument("--trec", action="store_true", help="Run WSLC on the TREC dataset")
    parser.add_argument("--trec-root", default="datasets/real-world/trec", help="directory containing trec data files")
    parser.add_argument("--trec-max-workers", type=int, default=500, help="Optional worker limit for TREC")
    parser.add_argument("--csv-answer", help="Path to answer.csv for generic datasets")
    parser.add_argument("--csv-truth", help="Path to truth.csv for generic datasets")
    parser.add_argument("--neighbors", type=int, help="Number of similar workers used by WSLC (default: all)")
    parser.add_argument("--aggregator", choices=["mv", "tiremge"], default="mv", help="Aggregator used for evaluation")
    parser.add_argument("--tiremge-few-shot", type=float, default=0.5)
    parser.add_argument("--tiremge-few-shot-seed", type=int, default=42)
    parser.add_argument("--tiremge-supervised-loss-weight", type=float, default=1.0)
    parser.add_argument("--tiremge-supervised-reliability-boost", type=float, default=0.0)
    parser.add_argument("--tiremge-object-supervision-strength", type=float, default=0.0)
    parser.add_argument("--tiremge-steps", type=int, default=200)
    parser.add_argument("--tiremge-learning-rate", type=float, default=1e-2)
    parser.add_argument("--tiremge-log-interval", type=int, default=10)
    parser.add_argument(
        "--embedding-mode",
        choices=["manual", "graphsage"],
        default="manual",
        help="Worker feature construction: manual=WSLC defaults, graphsage=bipartite GraphSAGE mean aggregation",
    )
    parser.add_argument(
        "--graphsage-dim",
        type=int,
        help="Embedding dimension for GraphSAGE worker vectors (default: num attributes)",
    )
    parser.add_argument(
        "--graphsage-layers",
        type=int,
        default=2,
        help="Number of mean-aggregation layers for GraphSAGE worker features",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    aggregator_opts: Optional[Dict[str, float]] = None
    if args.aggregator == "tiremge":
        aggregator_opts = {
            "tiremge_few_shot_ratio": args.tiremge_few_shot,
            "tiremge_few_shot_seed": args.tiremge_few_shot_seed,
            "tiremge_supervised_loss_weight": args.tiremge_supervised_loss_weight,
            "tiremge_supervised_reliability_boost": args.tiremge_supervised_reliability_boost,
            "tiremge_object_supervision_strength": args.tiremge_object_supervision_strength,
            "tiremge_steps": args.tiremge_steps,
            "tiremge_learning_rate": args.tiremge_learning_rate,
            "tiremge_log_interval": args.tiremge_log_interval,
        }

    ran_any = False
    if args.labelme:
        dataset = load_labelme_dataset(Path(args.labelme_root))
        _run_dataset(
            dataset,
            args.neighbors,
            args.aggregator,
            aggregator_opts,
            args.embedding_mode,
            args.graphsage_dim,
            args.graphsage_layers,
        )
        ran_any = True
    if args.trec:
        max_workers = args.trec_max_workers if args.trec_max_workers > 0 else None
        dataset = load_trec_dataset(Path(args.trec_root), max_workers=max_workers)
        _run_dataset(
            dataset,
            args.neighbors,
            args.aggregator,
            aggregator_opts,
            args.embedding_mode,
            args.graphsage_dim,
            args.graphsage_layers,
        )
        ran_any = True
    if args.csv_answer or args.csv_truth:
        if not args.csv_answer or not args.csv_truth:
            parser.error("Both --csv-answer and --csv-truth must be specified")
        dataset = load_csv_dataset(Path(args.csv_answer), Path(args.csv_truth))
        _run_dataset(
            dataset,
            args.neighbors,
            args.aggregator,
            aggregator_opts,
            args.embedding_mode,
            args.graphsage_dim,
            args.graphsage_layers,
        )
        ran_any = True
    if not ran_any:
        parser.error("No dataset selected. Use --labelme, --trec, or --csv-*. ")


if __name__ == "__main__":
    main()
