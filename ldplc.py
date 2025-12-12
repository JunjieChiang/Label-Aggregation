"""Python translation of the CEKA LDPLC implementation."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass
class Label:
    label_id: Optional[str]
    value: int
    example_id: str
    worker_id: str

    def getValue(self) -> int:
        return self.value


@dataclass
class Example:
    example_id: str
    attributes: List[float]
    true_label: Optional[Label] = None
    noisy_labels: Dict[str, Label] = field(default_factory=dict)

    def getId(self) -> str:
        return self.example_id

    def setTrueLabel(self, label: Label) -> None:
        self.true_label = label

    def getTrueLabel(self) -> Optional[Label]:
        return self.true_label

    def numAttributes(self) -> int:
        return len(self.attributes)

    def value(self, index: int) -> float:
        return self.attributes[index]

    def getNoisyLabelByWorkerId(self, worker_id: str) -> Optional[Label]:
        return self.noisy_labels.get(worker_id)

    def addNoisyLabel(self, label: Label) -> None:
        self.noisy_labels[label.worker_id] = label


@dataclass
class NoisyLabelSet:
    labels: List[Label] = field(default_factory=list)

    def getLabelSetSize(self) -> int:
        return len(self.labels)

    def getLabel(self, index: int) -> Label:
        return self.labels[index]

    def add(self, label: Label) -> None:
        self.labels.append(label)


@dataclass
class Worker:
    worker_id: str
    label_sets: List[NoisyLabelSet] = field(default_factory=lambda: [NoisyLabelSet()])

    def getId(self) -> str:
        return self.worker_id

    def addNoisyLabel(self, label: Label) -> None:
        self.label_sets[0].add(label)

    def getMultipleNoisyLabelSet(self, index: int) -> NoisyLabelSet:
        return self.label_sets[index]


@dataclass
class Category:
    category_id: str
    name: str

    def copy(self) -> Category:
        return Category(self.category_id, self.name)


@dataclass
class AttributeMeta:
    name: str
    is_nominal: bool = False
    values: Optional[List[str]] = None

    def copy(self) -> AttributeMeta:
        return AttributeMeta(self.name, self.is_nominal, list(self.values) if self.values else None)


@dataclass
class Dataset:
    attribute_info: List[AttributeMeta]
    examples: List[Example] = field(default_factory=list)
    workers: List[Worker] = field(default_factory=list)
    categories: List[Category] = field(default_factory=list)

    def generateEmpty(self) -> Dataset:
        return Dataset([meta.copy() for meta in self.attribute_info])

    def getExampleSize(self) -> int:
        return len(self.examples)

    def getWorkerSize(self) -> int:
        return len(self.workers)

    def getCategorySize(self) -> int:
        return len(self.categories)

    def numAttributes(self) -> int:
        return len(self.attribute_info)

    def numClasses(self) -> int:
        return len(self.categories)

    def getExampleByIndex(self, index: int) -> Example:
        return self.examples[index]

    def getWorkerByIndex(self, index: int) -> Worker:
        return self.workers[index]

    def getCategory(self, index: int) -> Category:
        return self.categories[index]

    def addExample(self, example: Example) -> None:
        self.examples.append(example)

    def addWorker(self, worker: Worker) -> None:
        self.workers.append(worker)

    def addCategory(self, category: Category) -> None:
        self.categories.append(category)


class EuclideanDistance:
    def __init__(self) -> None:
        self.dataset: Optional[Dataset] = None

    def setInstances(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def distance(self, example_one: Example, example_two: Example) -> float:
        limit = example_one.numAttributes() - 1
        total = 0.0
        for idx in range(limit):
            diff = example_one.value(idx) - example_two.value(idx)
            total += diff * diff
        return math.sqrt(total)


class _SimpleProgress:
    def __init__(self, total: int, desc: Optional[str] = None) -> None:
        self.total = max(total, 0)
        self.count = 0
        self.desc = desc or ""
        self.last_percent = -1

    def update(self) -> None:
        if self.total <= 0:
            return
        self.count += 1
        percent = int((self.count / self.total) * 100)
        if percent == self.last_percent:
            return
        self.last_percent = percent
        bar_length = 30
        filled = int(bar_length * self.count / self.total)
        bar = "#" * filled + "-" * (bar_length - filled)
        prefix = f"{self.desc}: " if self.desc else ""
        sys.stdout.write(f"\r{prefix}[{bar}] {percent:3d}%")
        if self.count >= self.total:
            sys.stdout.write("\n")
        sys.stdout.flush()


def iter_with_progress(iterable: Iterable[int], total: Optional[int] = None, desc: Optional[str] = None):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    if total is not None and total > 0:
        if tqdm is not None:
            return tqdm(iterable, total=total, desc=desc, leave=False)
        return _basic_progress(iterable, total, desc)
    return iterable


def _basic_progress(iterable: Iterable[int], total: int, desc: Optional[str]) -> Iterable[int]:
    progress = _SimpleProgress(total, desc)

    def generator():
        for item in iterable:
            yield item
            progress.update()

    return generator()


def dataset_copy(dataset: Dataset) -> Dataset:
    clone = dataset.generateEmpty()
    for category in dataset.categories:
        clone.addCategory(category.copy())
    for example in dataset.examples:
        clone.addExample(example)
    for worker in dataset.workers:
        new_worker = Worker(worker.getId())
        label_set = worker.getMultipleNoisyLabelSet(0)
        for idx in range(label_set.getLabelSetSize()):
            new_worker.addNoisyLabel(label_set.getLabel(idx))
        clone.addWorker(new_worker)
    return clone


def _parse_arff(arff_path: Path) -> Tuple[List[AttributeMeta], List[List[float]]]:
    attribute_info: List[AttributeMeta] = []
    value_maps: List[Optional[Dict[str, int]]] = []
    data_values: List[List[float]] = []
    data_mode = False

    with arff_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lower_line = line.lower()
            if lower_line.startswith("@relation"):
                continue
            if lower_line.startswith("@attribute"):
                parts = raw_line.strip().split(None, 2)
                if len(parts) < 3:
                    continue
                name = parts[1].strip("'\"")
                type_part = parts[2].strip()
                type_lower = type_part.lower()
                if "{" in type_part and "}" in type_part:
                    brace_start = type_part.find("{")
                    brace_end = type_part.rfind("}")
                    if brace_end <= brace_start:
                        continue
                    values_str = type_part[brace_start + 1 : brace_end]
                    values = [
                        token.strip().strip("'\"")
                        for token in values_str.split(",")
                        if token.strip()
                    ]
                    attribute_info.append(AttributeMeta(name, True, values))
                    value_maps.append({value: idx for idx, value in enumerate(values)})
                elif type_lower in {"numeric", "real", "integer"}:
                    attribute_info.append(AttributeMeta(name))
                    value_maps.append(None)
                else:
                    attribute_info.append(AttributeMeta(name))
                    value_maps.append(None)
                continue
            if lower_line.startswith("@data"):
                data_mode = True
                continue
            if not data_mode:
                continue
            tokens = [token.strip() for token in line.split(",")]
            if len(tokens) != len(attribute_info):
                continue
            converted: List[float] = []
            for idx, token in enumerate(tokens):
                attr = attribute_info[idx]
                token_clean = token.strip("'\"")
                if attr.is_nominal:
                    value_map = value_maps[idx]
                    if value_map is None:
                        value_map = {}
                        value_maps[idx] = value_map
                    if token_clean in ("?", ""):
                        numeric_value = 0.0
                    else:
                        numeric_value = float(value_map.get(token_clean, 0))
                else:
                    if token_clean in ("?", ""):
                        numeric_value = 0.0
                    else:
                        try:
                            numeric_value = float(token_clean)
                        except ValueError:
                            numeric_value = 0.0
                converted.append(numeric_value)
            data_values.append(converted)
    return attribute_info, data_values


def _load_gold_labels(gold_path: Path) -> Tuple[Dict[int, int], int]:
    gold: Dict[int, int] = {}
    max_label = -1
    with gold_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                example_idx = int(parts[0])
                label_value = int(parts[1])
            except ValueError:
                continue
            gold[example_idx] = label_value
            if label_value > max_label:
                max_label = label_value
    num_classes = max_label + 1 if max_label >= 0 else 0
    return gold, num_classes


def _load_worker_annotations(dataset: Dataset, response_path: Path) -> None:
    workers: Dict[str, Worker] = {worker.getId(): worker for worker in dataset.workers}
    examples = dataset.examples
    with response_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            worker_id, example_id_text, label_text = parts[:3]
            try:
                example_idx = int(example_id_text)
                label_value = int(label_text)
            except ValueError:
                continue
            if example_idx < 0 or example_idx >= len(examples):
                continue
            worker = workers.get(worker_id)
            if worker is None:
                worker = Worker(worker_id)
                dataset.addWorker(worker)
                workers[worker_id] = worker
            example = examples[example_idx]
            label = Label(None, label_value, example.getId(), worker_id)
            example.addNoisyLabel(label)
            worker.addNoisyLabel(label)


def _load_truth_csv(truth_path: Path) -> Tuple[Dict[str, int], int]:
    truth: Dict[str, int] = {}
    max_label = -1
    with truth_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            object_id = row[0].strip()
            label_text = row[1].strip()
            if not object_id or not label_text:
                continue
            try:
                label_value = int(label_text)
            except ValueError:
                continue
            truth[object_id] = label_value
            if label_value > max_label:
                max_label = label_value
    return truth, max_label


def _sort_object_ids(object_ids: Iterable[str]) -> List[str]:
    def sort_key(value: str) -> Tuple[int, object]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted((str(value) for value in object_ids), key=sort_key)


def _majority_vote(example: Example, num_classes: int) -> int:
    if num_classes == 0:
        return 0
    counts = [0 for _ in range(num_classes)]
    for label in example.noisy_labels.values():
        counts[label.getValue()] += 1
    if not any(counts):
        return 0
    return max_index(counts)


def evaluate_majority_vote(dataset: Dataset) -> float:
    total = 0
    correct = 0
    num_classes = dataset.numClasses()
    for example in dataset.examples:
        true_label = example.getTrueLabel()
        if true_label is None:
            continue
        prediction = _majority_vote(example, num_classes)
        total += 1
        if prediction == true_label.getValue():
            correct += 1
    return correct / total if total else 0.0


def load_labelme_dataset(root: Path) -> Dataset:
    arff_path = root / "LabelMe.arff"
    response_path = root / "LabelMe.response.txt"
    gold_path = root / "LabelMe.gold.txt"

    attribute_info, data_values = _parse_arff(arff_path)
    gold_labels, inferred_classes = _load_gold_labels(gold_path)

    dataset = Dataset(attribute_info)
    if attribute_info:
        class_attr = attribute_info[-1]
    else:
        class_attr = AttributeMeta("class", True, [])
        attribute_info.append(class_attr)
    if class_attr.is_nominal and class_attr.values:
        category_names = class_attr.values
    else:
        num_classes = inferred_classes if inferred_classes else max(gold_labels.values(), default=0) + 1
        if num_classes <= 0:
            num_classes = 1
        category_names = [str(i) for i in range(num_classes)]
        class_attr.is_nominal = True
        class_attr.values = category_names
    for idx, name in enumerate(category_names):
        dataset.addCategory(Category(str(idx), name))

    for idx, values in enumerate(data_values):
        example_id = str(idx)
        example_values = list(values)
        example = Example(example_id, example_values)
        gold_value = gold_labels.get(idx)
        if gold_value is not None:
            example.setTrueLabel(Label(None, gold_value, example_id, "truth"))
            if example_values:
                example_values[-1] = float(gold_value)
        dataset.addExample(example)

    _load_worker_annotations(dataset, response_path)
    return dataset


def load_trec_dataset(root: Path, max_workers: Optional[int] = None) -> Dataset:
    response_path = root / "trec_responses.txt"
    gold_path = root / "trec_truth.txt"

    gold_labels_raw, _ = _load_gold_labels(gold_path)
    worker_annotations: Dict[str, List[Tuple[int, int]]] = {}
    max_example_idx = max(gold_labels_raw.keys(), default=-1)
    label_values = set(gold_labels_raw.values())
    with response_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                example_idx = int(parts[0])
                worker_idx = int(parts[1])
                label_value = int(parts[2])
            except ValueError:
                continue
            worker_id = str(worker_idx)
            worker_annotations.setdefault(worker_id, []).append((example_idx, label_value))
            if example_idx > max_example_idx:
                max_example_idx = example_idx
            label_values.add(label_value)

    num_examples = max_example_idx + 1 if max_example_idx >= 0 else len(gold_labels_raw)
    if not label_values:
        label_values = {0}
    sorted_labels = sorted(label_values)
    label_mapping = {value: idx for idx, value in enumerate(sorted_labels)}
    gold_labels = {idx: label_mapping[label] for idx, label in gold_labels_raw.items()}
    num_classes = len(sorted_labels)
    class_names = [str(value) for value in sorted_labels]

    attribute_info = [
        AttributeMeta("example_index"),
        AttributeMeta("class", is_nominal=True, values=list(class_names)),
    ]
    dataset = Dataset(attribute_info)
    for idx in range(num_classes):
        dataset.addCategory(Category(str(idx), class_names[idx]))

    for idx in range(num_examples):
        attrs = [float(idx), 0.0]
        example = Example(str(idx), attrs)
        gold_value = gold_labels.get(idx)
        if gold_value is not None:
            example.setTrueLabel(Label(None, gold_value, example.getId(), "truth"))
            example.attributes[-1] = float(gold_value)
        dataset.addExample(example)

    selected_workers: List[Tuple[str, List[Tuple[int, int]]]]
    if max_workers is not None and max_workers > 0 and len(worker_annotations) > max_workers:
        sorted_workers = sorted(worker_annotations.items(), key=lambda item: len(item[1]), reverse=True)
        selected_workers = sorted_workers[:max_workers]
    else:
        selected_workers = list(worker_annotations.items())

    for worker_id, annotations in selected_workers:
        worker = Worker(worker_id)
        dataset.addWorker(worker)
        for example_idx, label_value in annotations:
            if example_idx < 0 or example_idx >= dataset.getExampleSize():
                continue
            mapped_value = label_mapping.get(label_value)
            if mapped_value is None:
                continue
            example = dataset.getExampleByIndex(example_idx)
            label = Label(None, mapped_value, example.getId(), worker_id)
            example.addNoisyLabel(label)
            worker.addNoisyLabel(label)

    return dataset


def load_csv_dataset(answer_path: Path, truth_path: Path) -> Dataset:
    if not answer_path.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    truth_labels, max_truth_label = _load_truth_csv(truth_path)
    worker_annotations: Dict[str, List[Tuple[str, int]]] = {}
    object_ids = set(truth_labels.keys())
    max_label_value = max_truth_label

    with answer_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            object_id = row[0].strip()
            worker_id = row[1].strip()
            label_text = row[2].strip()
            if not object_id or not worker_id or not label_text:
                continue
            try:
                label_value = int(label_text)
            except ValueError:
                continue
            worker_annotations.setdefault(worker_id, []).append((object_id, label_value))
            object_ids.add(object_id)
            if label_value > max_label_value:
                max_label_value = label_value

    num_classes = max(max_label_value + 1, 1)
    category_names = [str(i) for i in range(num_classes)]
    attribute_info = [
        AttributeMeta("example_index"),
        AttributeMeta("class", is_nominal=True, values=list(category_names)),
    ]

    dataset = Dataset(attribute_info)
    for idx, name in enumerate(category_names):
        dataset.addCategory(Category(str(idx), name))

    sorted_objects = _sort_object_ids(object_ids)
    object_index: Dict[str, int] = {}
    for idx, object_id in enumerate(sorted_objects):
        attrs = [float(idx), 0.0]
        example = Example(object_id, attrs)
        gold_value = truth_labels.get(object_id)
        if gold_value is not None:
            example.setTrueLabel(Label(None, gold_value, example.getId(), "truth"))
            example.attributes[-1] = float(gold_value)
        dataset.addExample(example)
        object_index[object_id] = idx

    for worker_id, annotations in worker_annotations.items():
        worker = Worker(worker_id)
        dataset.addWorker(worker)
        for object_id, label_value in annotations:
            idx = object_index.get(object_id)
            if idx is None:
                continue
            example = dataset.getExampleByIndex(idx)
            label = Label(None, label_value, example.getId(), worker_id)
            example.addNoisyLabel(label)
            worker.addNoisyLabel(label)

    return dataset


def _write_dataset_to_csv(dataset: Dataset, answer_path: Path, truth_path: Path) -> None:
    with answer_path.open("w", newline="", encoding="utf-8") as answer_file:
        writer = csv.writer(answer_file)
        for example in dataset.examples:
            for label in example.noisy_labels.values():
                writer.writerow([example.getId(), label.worker_id, label.getValue()])
    with truth_path.open("w", newline="", encoding="utf-8") as truth_file:
        writer = csv.writer(truth_file)
        for example in dataset.examples:
            true_label = example.getTrueLabel()
            if true_label is None:
                continue
            writer.writerow([example.getId(), true_label.getValue()])


def _evaluate_with_tiremge(dataset: Dataset, options: Dict[str, Any]) -> float:
    try:
        import demo as tiremge_demo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TiReMGE aggregator requires TensorFlow and related dependencies") from exc

    few_shot_ratio = float(options.get("tiremge_few_shot_ratio", 0.5))
    steps = int(options.get("tiremge_steps", 200))
    learning_rate = float(options.get("tiremge_learning_rate", 1e-2))
    log_interval = int(options.get("tiremge_log_interval", 10))
    few_shot_seed = int(options.get("tiremge_few_shot_seed", 42))
    supervised_loss_weight = float(options.get("tiremge_supervised_loss_weight", 1.0))
    supervised_reliability_boost = float(options.get("tiremge_supervised_reliability_boost", 0.0))
    object_supervision_strength = float(options.get("tiremge_object_supervision_strength", 0.0))

    with tempfile.TemporaryDirectory(prefix="ldplc_tiremge_") as tmpdir:
        tmp_dataset_dir = Path(tmpdir) / "dataset"
        tmp_dataset_dir.mkdir(parents=True, exist_ok=True)
        answer_path = tmp_dataset_dir / "answer.csv"
        truth_path = tmp_dataset_dir / "truth.csv"
        _write_dataset_to_csv(dataset, answer_path, truth_path)
        result = tiremge_demo.run_tiremge(
            answer_path=str(answer_path),
            truth_path=str(truth_path),
            dataset_name="ldplc_temp",
            metrics_dir=None,
            few_shot_ratio=few_shot_ratio,
            supervised_loss_weight=supervised_loss_weight,
            supervised_reliability_boost=supervised_reliability_boost,
            object_supervision_strength=object_supervision_strength,
            few_shot_seed=few_shot_seed,
            log_data_sample_interval=log_interval,
            steps=steps,
            learning_rate=learning_rate,
            save_outputs=False,
        )
    return float(result.get("best_accuracy", 0.0))


def _evaluate_dataset(dataset: Dataset, aggregator: str, options: Optional[Dict[str, Any]] = None) -> float:
    opts = options or {}
    if aggregator == "mv":
        return evaluate_majority_vote(dataset)
    if aggregator == "tiremge":
        return _evaluate_with_tiremge(dataset, opts)
    raise ValueError(f"Unsupported aggregator: {aggregator}")


def _aggregator_display(name: str) -> str:
    normalized = name.lower()
    if normalized == "mv":
        return "MV"
    if normalized == "tiremge":
        return "TiReMGE"
    return name


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize(values: List[float]) -> None:
    total = sum(values)
    if total > 0:
        for idx, value in enumerate(values):
            values[idx] = value / total


def max_index(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def min_index(values: Sequence[float]) -> int:
    return min(range(len(values)), key=lambda idx: values[idx])


class WorkerSimilarity:
    def __init__(self) -> None:
        self.m_numClasses = 0
        self.m_numWorkers = 0
        self.m_numAttributes = 0

    def doInference(self, dataset: Dataset) -> List[List[float]]:
        self.m_numClasses = dataset.numClasses()
        self.m_numWorkers = dataset.getWorkerSize()
        self.m_numAttributes = dataset.numAttributes()
        if self.m_numWorkers == 0 or self.m_numAttributes <= 1:
            return []

        sub_datasets = [dataset.generateEmpty() for _ in range(self.m_numWorkers)]
        for example in dataset.examples:
            for w, worker in enumerate(dataset.workers):
                if example.getNoisyLabelByWorkerId(worker.getId()) is not None:
                    sub_datasets[w].addExample(example)

        attribute_vectors = [
            self._calc_attribute_vector(sub_datasets[w], dataset.getWorkerByIndex(w).getId(), dataset.attribute_info)
            for w in range(self.m_numWorkers)
        ]

        similarity = [[0.0 for _ in range(self.m_numWorkers)] for _ in range(self.m_numWorkers)]
        for i in range(self.m_numWorkers):
            for j in range(self.m_numWorkers):
                similarity[i][j] = 0.0 if i == j else self._calc_similarity(attribute_vectors[i], attribute_vectors[j])
        return similarity

    @staticmethod
    def _calc_similarity(vec_one: Sequence[float], vec_two: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(vec_one, vec_two))
        denom_one = math.sqrt(sum(a * a for a in vec_one))
        denom_two = math.sqrt(sum(b * b for b in vec_two))
        if numerator == 0 or denom_one == 0 or denom_two == 0:
            return 0.0
        cosine = numerator / (denom_one * denom_two)
        return 0.5 * (1.0 + cosine)

    def _calc_attribute_vector(self, data: Dataset, worker_id: str, attribute_info: List[AttributeMeta]) -> List[float]:
        num_dataset = data.getExampleSize()
        result = [0.0 for _ in range(self.m_numAttributes - 1)]
        if num_dataset == 0:
            return result

        pc = [0.0 for _ in range(self.m_numClasses)]
        classy = [[0.0 for _ in range(num_dataset)] for _ in range(self.m_numClasses)]
        class_labels: List[int] = []

        for idx in range(num_dataset):
            example = data.getExampleByIndex(idx)
            label = example.getNoisyLabelByWorkerId(worker_id)
            class_idx = label.getValue() if label else 0
            class_labels.append(class_idx)
            pc[class_idx] += 1.0
            classy[class_idx][idx] = 1.0

        normalize(pc)

        for att in range(self.m_numAttributes - 1):
            meta = attribute_info[att]
            if meta.is_nominal:
                num_att = len(meta.values) if meta.values else 0
                observed_values: List[int] = []
                for idx in range(num_dataset):
                    value = int(round(data.getExampleByIndex(idx).value(att)))
                    observed_values.append(value)
                    num_att = max(num_att, value + 1)
                attx = [[0.0 for _ in range(num_dataset)] for _ in range(num_att)]
                paic = [[0.0 for _ in range(self.m_numClasses)] for _ in range(num_att)]
                for idx, value in enumerate(observed_values):
                    value = max(0, min(value, num_att - 1))
                    class_idx = class_labels[idx]
                    paic[value][class_idx] += 1.0
                    attx[value][idx] = 1.0
                for q in range(self.m_numClasses):
                    for p in range(num_att):
                        paic[p][q] /= num_dataset
                        if paic[p][q] != 0:
                            result[att] += paic[p][q] * self._calc_attribute_r(attx[p], classy[q])
            else:
                att_values = [data.getExampleByIndex(idx).value(att) for idx in range(num_dataset)]
                for q in range(self.m_numClasses):
                    if pc[q] != 0:
                        result[att] += pc[q] * self._calc_attribute_r(att_values, classy[q])
        return result

    @staticmethod
    def _calc_attribute_r(x: Sequence[float], y: Sequence[float]) -> float:
        mean_x = mean(x)
        mean_y = mean(y)
        numerator = 0.0
        denom_x = 0.0
        denom_y = 0.0
        for xi, yi in zip(x, y):
            numerator += (xi - mean_x) * (yi - mean_y)
            denom_x += (xi - mean_x) ** 2
            denom_y += (yi - mean_y) ** 2
        if denom_x == 0 or denom_y == 0:
            return 0.0
        return numerator / math.sqrt(denom_x * denom_y)


class LDPLC:
    def __init__(
        self,
        k: int = 5,
        iterations: int = 5,
        qp_solver: Optional[Callable[[Sequence[Sequence[float]], Sequence[float], Sequence[Sequence[float]], Sequence[float], Sequence[Sequence[float]], Sequence[float], Sequence[float], Sequence[float]], Sequence[float]]] = None,
        distance_function: Optional[EuclideanDistance] = None,
        num_threads: int = 1,
    ) -> None:
        self.m_K = k
        self.iter = iterations
        self.m_qp_solver = qp_solver
        self.m_DistanceFunction = distance_function or EuclideanDistance()
        self.m_numExamples = 0
        self.m_numWorkers = 0
        self.m_numClasses = 0
        self.m_numThreads = max(1, num_threads)

    def SetIter(self, iterations: int) -> None:
        self.iter = iterations

    def SetK(self, k: int) -> None:
        self.m_K = k

    def SetQP(self, solver: Callable[..., Sequence[float]]) -> None:
        self.m_qp_solver = solver

    def doInference(self, dataset: Dataset) -> Dataset:
        self.m_DistanceFunction.setInstances(dataset)
        self.m_numExamples = dataset.getExampleSize()
        self.m_numWorkers = dataset.getWorkerSize()
        self.m_numClasses = dataset.numClasses()

        if self.m_numExamples == 0 or self.m_numWorkers == 0:
            return dataset

        def _parallel_for(count: int, func: Callable[[int], None]) -> None:
            if self.m_numThreads <= 1 or count <= 1:
                for idx in range(count):
                    func(idx)
            else:
                with ThreadPoolExecutor(max_workers=self.m_numThreads) as executor:
                    list(executor.map(func, range(count)))

        def _run_over_examples(desc: Optional[str], func: Callable[[int], None], show_progress: bool = True) -> None:
            if self.m_numThreads <= 1:
                iterator: Iterable[int]
                if show_progress:
                    iterator = iter_with_progress(range(self.m_numExamples), total=self.m_numExamples, desc=desc)
                else:
                    iterator = range(self.m_numExamples)
                for idx in iterator:
                    func(idx)
            else:
                if desc and show_progress:
                    print(f"{desc} (parallel x{self.m_numThreads})")
                _parallel_for(self.m_numExamples, func)

        worker_similarity = WorkerSimilarity().doInference(dataset_copy(dataset))
        distributes = [
            [[0.0 for _ in range(self.m_numClasses)] for _ in range(self.m_numWorkers)]
            for _ in range(self.m_numExamples)
        ]
        knearest = [[0 for _ in range(self.m_K)] for _ in range(self.m_numExamples)]
        knearest_weight = [[0.0 for _ in range(self.m_K)] for _ in range(self.m_numExamples)]

        def _init_example(i: int) -> None:
            example = dataset.getExampleByIndex(i)
            for r in range(self.m_numWorkers):
                worker = dataset.getWorkerByIndex(r)
                label = example.getNoisyLabelByWorkerId(worker.getId())
                if label is None:
                    distribute = [0.0 for _ in range(self.m_numClasses)]
                    for k in range(self.m_numWorkers):
                        neighbour = dataset.getWorkerByIndex(k)
                        neighbour_label = example.getNoisyLabelByWorkerId(neighbour.getId())
                        if neighbour_label is None:
                            continue
                        distribute[neighbour_label.getValue()] += worker_similarity[r][k]
                    if mean(distribute) != 0:
                        normalize(distribute)
                    distributes[i][r] = distribute
                else:
                    distribute = [0.0 for _ in range(self.m_numClasses)]
                    distribute[label.getValue()] = 1.0
                    distributes[i][r] = distribute

        _run_over_examples("Initializing", _init_example)

        if self.m_K > 0:
            def _compute_knn(i: int) -> None:
                knearest[i] = self._find_knearest(dataset, i)
                g_matrix = self._calculate_g(dataset, knearest[i], i)
                knearest_weight[i] = list(
                    self._solve_qp(
                        g_matrix,
                        [0.0] * self.m_K,
                        [[0.0] * self.m_K],
                        [0.0],
                        [[1.0] * self.m_K],
                        [1.0],
                        [0.0] * self.m_K,
                        [1.0] * self.m_K,
                    )
                )

            _run_over_examples("Computing kNN", _compute_knn)

        old_distributes = [
            [list(distributes[i][r]) for r in range(self.m_numWorkers)]
            for i in range(self.m_numExamples)
        ]
        new_distributes = [
            [list(distributes[i][r]) for r in range(self.m_numWorkers)]
            for i in range(self.m_numExamples)
        ]

        for _ in iter_with_progress(range(self.iter), total=self.iter, desc="Propagating"):
            def _update_example(i: int) -> None:
                example = dataset.getExampleByIndex(i)
                for r in range(self.m_numWorkers):
                    worker = dataset.getWorkerByIndex(r)
                    if example.getNoisyLabelByWorkerId(worker.getId()) is None:
                        for c in range(self.m_numClasses):
                            updated = 0.5 * distributes[i][r][c]
                            for j in range(self.m_K):
                                updated += 0.5 * knearest_weight[i][j] * old_distributes[knearest[i][j]][r][c]
                            new_distributes[i][r][c] = updated

            _run_over_examples(None, _update_example, show_progress=False)
            for i in range(self.m_numExamples):
                for r in range(self.m_numWorkers):
                    for c in range(self.m_numClasses):
                        old_distributes[i][r][c] = new_distributes[i][r][c]

        def _assign_labels(i: int) -> None:
            example = dataset.getExampleByIndex(i)
            for r in range(self.m_numWorkers):
                worker = dataset.getWorkerByIndex(r)
                if example.getNoisyLabelByWorkerId(worker.getId()) is None:
                    weights = new_distributes[i][r]
                    value = max_index(weights)
                    label = Label(None, value, example.getId(), worker.getId())
                    example.addNoisyLabel(label)
                    worker.addNoisyLabel(label)

        _run_over_examples("Assigning labels", _assign_labels)
        return dataset

    def _find_knearest(self, dataset: Dataset, index: int) -> List[int]:
        example = dataset.getExampleByIndex(index)
        distances = [0.0 for _ in range(self.m_numExamples)]
        for i in range(self.m_numExamples):
            if i == index:
                distances[i] = float("inf")
            else:
                other = dataset.getExampleByIndex(i)
                distances[i] = self.m_DistanceFunction.distance(example, other)
        indices = []
        for _ in range(self.m_K):
            neighbour = min_index(distances)
            indices.append(neighbour)
            distances[neighbour] = float("inf")
        return indices

    def _calculate_g(self, dataset: Dataset, knearest: Sequence[int], index: int) -> List[List[float]]:
        example = dataset.getExampleByIndex(index)
        num_attributes = example.numAttributes() - 1
        g_matrix = [[0.0 for _ in range(self.m_K)] for _ in range(self.m_K)]
        for i in range(self.m_K):
            example_one = dataset.getExampleByIndex(knearest[i])
            for j in range(i, self.m_K):
                example_two = dataset.getExampleByIndex(knearest[j])
                value = 0.0
                for k in range(num_attributes):
                    diff_one = example.value(k) - example_one.value(k)
                    diff_two = example.value(k) - example_two.value(k)
                    value += diff_one * diff_two
                g_matrix[i][j] = value
                g_matrix[j][i] = value
        return g_matrix

    def _solve_qp(
        self,
        h_matrix: Sequence[Sequence[float]],
        f_vector: Sequence[float],
        a_matrix: Sequence[Sequence[float]],
        b_vector: Sequence[float],
        aeq_matrix: Sequence[Sequence[float]],
        beq_vector: Sequence[float],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> Sequence[float]:
        if self.m_qp_solver is not None:
            return self.m_qp_solver(h_matrix, f_vector, a_matrix, b_vector, aeq_matrix, beq_vector, lower_bounds, upper_bounds)
        if np is None:
            return [1.0 / len(h_matrix) for _ in range(len(h_matrix))]
        h = np.asarray(h_matrix, dtype=float)
        identity = np.eye(len(h)) * 1e-6
        ones = np.ones(len(h))
        try:
            weights = np.linalg.solve(h + identity, ones)
        except np.linalg.LinAlgError:
            weights = ones
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights /= total
        return weights.tolist()


def run_labelme(
    root: str,
    k: int = 5,
    iterations: int = 5,
    aggregator: str = "mv",
    aggregator_opts: Optional[Dict[str, Any]] = None,
    num_threads: int = 1,
) -> None:
    """Run LDPLC on the real LabelMe dataset and report majority-vote accuracy."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"LabelMe root directory not found: {root_path}")
    dataset = load_labelme_dataset(root_path)
    print(
        f"Loaded LabelMe dataset from {root_path} "
        f"(examples={dataset.getExampleSize()}, workers={dataset.getWorkerSize()}, "
        f"classes={dataset.numClasses()})"
    )
    label = _aggregator_display(aggregator)
    baseline_accuracy = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Baseline {label} accuracy (before LDPLC): {baseline_accuracy * 100:.2f}%")

    algorithm = LDPLC(k=k, iterations=iterations, num_threads=num_threads)
    algorithm.doInference(dataset)
    post_accuracy = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Post-LDPLC {label} accuracy: {post_accuracy * 100:.2f}%")


def run_trec(
    root: str,
    k: int = 5,
    iterations: int = 5,
    max_workers: Optional[int] = None,
    aggregator: str = "mv",
    aggregator_opts: Optional[Dict[str, Any]] = None,
    num_threads: int = 1,
) -> None:
    """Run LDPLC on the TREC dataset and report majority-vote accuracy."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"TREC root directory not found: {root_path}")
    dataset = load_trec_dataset(root_path, max_workers=max_workers)
    print(
        f"Loaded TREC dataset from {root_path} "
        f"(examples={dataset.getExampleSize()}, workers={dataset.getWorkerSize()}, "
        f"classes={dataset.numClasses()})"
    )
    label = _aggregator_display(aggregator)
    baseline_accuracy = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Baseline {label} accuracy (before LDPLC): {baseline_accuracy * 100:.2f}%")

    algorithm = LDPLC(k=k, iterations=iterations, num_threads=num_threads)
    algorithm.doInference(dataset)
    post_accuracy = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Post-LDPLC {label} accuracy: {post_accuracy * 100:.2f}%")


def run_csv(
    answer_path: str,
    truth_path: str,
    k: int = 5,
    iterations: int = 5,
    aggregator: str = "mv",
    aggregator_opts: Optional[Dict[str, Any]] = None,
    num_threads: int = 1,
) -> None:
    """Run LDPLC on a generic CSV dataset using answer.csv/truth.csv files."""
    answer = Path(answer_path)
    truth = Path(truth_path)
    dataset = load_csv_dataset(answer, truth)
    dataset_name = answer.parent.name if answer.parent != answer else answer.stem
    print(
        f"Loaded CSV dataset {dataset_name} "
        f"(examples={dataset.getExampleSize()}, workers={dataset.getWorkerSize()}, "
        f"classes={dataset.numClasses()})"
    )
    label = _aggregator_display(aggregator)
    baseline_accuracy = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Baseline {label} accuracy (before LDPLC): {baseline_accuracy * 100:.2f}%")

    algorithm = LDPLC(k=k, iterations=iterations, num_threads=num_threads)
    algorithm.doInference(dataset)
    post_accuracy = _evaluate_dataset(dataset, aggregator, aggregator_opts)
    print(f"Post-LDPLC {label} accuracy: {post_accuracy * 100:.2f}%")


def run_demo() -> None:
    """Small demo that mirrors the Java Test_S setup on a toy dataset."""
    attributes = [AttributeMeta("x1"), AttributeMeta("x2"), AttributeMeta("class", is_nominal=True, values=["0", "1"])]
    dataset = Dataset(attributes)
    dataset.addCategory(Category("0", "class-0"))
    dataset.addCategory(Category("1", "class-1"))

    examples = [
        Example("e0", [0.0, 1.0, 0.0]),
        Example("e1", [1.0, 0.5, 0.0]),
        Example("e2", [0.8, 1.5, 0.0]),
    ]
    true_values = [0, 1, 0]
    for example, true_value in zip(examples, true_values):
        example.setTrueLabel(Label(None, true_value, example.getId(), "truth"))
        dataset.addExample(example)

    workers = [Worker("w0"), Worker("w1"), Worker("w2")]
    for worker in workers:
        dataset.addWorker(worker)

    dataset.getExampleByIndex(0).addNoisyLabel(Label(None, 0, "e0", "w0"))
    workers[0].addNoisyLabel(Label(None, 0, "e0", "w0"))
    dataset.getExampleByIndex(1).addNoisyLabel(Label(None, 1, "e1", "w0"))
    workers[0].addNoisyLabel(Label(None, 1, "e1", "w0"))
    dataset.getExampleByIndex(2).addNoisyLabel(Label(None, 1, "e2", "w1"))
    workers[1].addNoisyLabel(Label(None, 1, "e2", "w1"))

    algorithm = LDPLC(k=2, iterations=3)
    completed_dataset = algorithm.doInference(dataset)

    for example in completed_dataset.examples:
        labels = {
            worker.getId(): label.getValue()
            for worker in completed_dataset.workers
            if (label := example.getNoisyLabelByWorkerId(worker.getId())) is not None
        }
        print(f"Example {example.getId()} labels: {labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LDPLC reference implementation")
    parser.add_argument("--demo", action="store_true", help="Run the toy LDPLC demo")
    parser.add_argument("--labelme", action="store_true", help="Run LDPLC on the LabelMe dataset")
    parser.add_argument(
        "--labelme-root",
        default="datasets/real-world/labelme",
        help="Directory containing LabelMe.arff/LabelMe.response.txt/LabelMe.gold.txt",
    )
    parser.add_argument("--trec", action="store_true", help="Run LDPLC on the TREC dataset")
    parser.add_argument(
        "--trec-root",
        default="datasets/real-world/trec",
        help="Directory containing trec_responses.txt/trec_truth.txt",
    )
    parser.add_argument(
        "--trec-max-workers",
        type=int,
        default=500,
        help="Limit number of TREC workers (0 for unlimited); limiting helps keep memory manageable",
    )
    parser.add_argument(
        "--csv-answer",
        help="Path to answer.csv (object_id, worker_id, label) for generic CSV datasets",
    )
    parser.add_argument(
        "--csv-truth",
        help="Path to truth.csv (object_id, label) for generic CSV datasets",
    )
    parser.add_argument(
        "--aggregator",
        choices=["mv", "tiremge"],
        default="mv",
        help="Aggregation model used for reporting accuracy",
    )
    parser.add_argument(
        "--tiremge-few-shot",
        type=float,
        default=0,
        help="Few-shot supervision ratio for the TiReMGE aggregator",
    )
    parser.add_argument(
        "--tiremge-few-shot-seed",
        type=int,
        default=42,
        help="Random seed controlling TiReMGE few-shot sampling",
    )
    parser.add_argument(
        "--tiremge-supervised-loss-weight",
        type=float,
        default=1.0,
        help="Supervised loss weight used by the TiReMGE aggregator",
    )
    parser.add_argument(
        "--tiremge-supervised-reliability-boost",
        type=float,
        default=0.0,
        help="Reliability boost applied to supervised workers for TiReMGE",
    )
    parser.add_argument(
        "--tiremge-object-supervision-strength",
        type=float,
        default=0.0,
        help="Object supervision strength used by TiReMGE",
    )
    parser.add_argument(
        "--tiremge-steps",
        type=int,
        default=200,
        help="Number of training steps for the TiReMGE aggregator",
    )
    parser.add_argument(
        "--tiremge-learning-rate",
        type=float,
        default=1e-2,
        help="Learning rate for the TiReMGE aggregator",
    )
    parser.add_argument(
        "--tiremge-log-interval",
        type=int,
        default=10,
        help="Interval between TiReMGE aggregator logging samples",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads used inside LDPLC for per-example work",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of neighbours used by LDPLC")
    parser.add_argument("--iterations", type=int, default=5, help="Number of propagation iterations")
    args = parser.parse_args()

    aggregator_opts: Optional[Dict[str, Any]] = None
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
        run_labelme(
            args.labelme_root,
            k=args.k,
            iterations=args.iterations,
            aggregator=args.aggregator,
            aggregator_opts=aggregator_opts,
            num_threads=args.threads,
        )
        ran_any = True
    if args.trec:
        max_workers = args.trec_max_workers if args.trec_max_workers > 0 else None
        run_trec(
            args.trec_root,
            k=args.k,
            iterations=args.iterations,
            max_workers=max_workers,
            aggregator=args.aggregator,
            aggregator_opts=aggregator_opts,
            num_threads=args.threads,
        )
        ran_any = True
    if args.csv_answer or args.csv_truth:
        if not args.csv_answer or not args.csv_truth:
            parser.error("Both --csv-answer and --csv-truth must be provided together.")
        run_csv(
            args.csv_answer,
            args.csv_truth,
            k=args.k,
            iterations=args.iterations,
            aggregator=args.aggregator,
            aggregator_opts=aggregator_opts,
            num_threads=args.threads,
        )
        ran_any = True
    if args.demo or not ran_any:
        run_demo()


if __name__ == "__main__":
    main()
