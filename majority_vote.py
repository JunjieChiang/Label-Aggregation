import argparse
import csv
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


def read_answers(answer_path: str) -> Dict[str, List[int]]:
    """Load each object's labels provided by workers."""
    votes: Dict[str, List[int]] = defaultdict(list)
    with open(answer_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 3:
                continue
            object_id, _, label = row[0].strip(), row[1].strip(), row[2].strip()
            if not object_id or not label:
                continue
            votes[object_id].append(int(label))
    return votes


def read_truth(truth_path: str) -> Dict[str, int]:
    """Load ground-truth labels for each object."""
    truth: Dict[str, int] = {}
    with open(truth_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            object_id, label = row[0].strip(), row[1].strip()
            if not object_id or not label:
                continue
            truth[object_id] = int(label)
    return truth


def majority_vote(labels: Iterable[int]) -> int:
    """Return the label with the highest vote count (ties broken by smaller label)."""
    counter = Counter(labels)
    if not counter:
        raise ValueError("majority_vote called with an empty label list.")
    max_votes = max(counter.values())
    candidates = [label for label, count in counter.items() if count == max_votes]
    return min(candidates)


def evaluate_dataset(dataset: str, data_root: str) -> Tuple[float, int, int]:
    """Compute MV accuracy for a dataset located under data_root/dataset."""
    dataset_dir = os.path.join(data_root, dataset)
    answer_path = os.path.join(dataset_dir, "answer.csv")
    truth_path = os.path.join(dataset_dir, "truth.csv")

    if not os.path.exists(answer_path):
        raise FileNotFoundError(f"Missing answer file: {answer_path}")
    if not os.path.exists(truth_path):
        raise FileNotFoundError(f"Missing truth file: {truth_path}")

    votes = read_answers(answer_path)
    truth = read_truth(truth_path)

    y_true: List[int] = []
    y_pred: List[int] = []
    skipped = 0

    for object_id, true_label in truth.items():
        labels = votes.get(object_id)
        if not labels:
            skipped += 1
            continue
        y_true.append(true_label)
        y_pred.append(majority_vote(labels))

    if not y_true:
        raise ValueError(f"No overlapping objects between answers and truths in dataset {dataset}.")

    correct = sum(int(p == t) for p, t in zip(y_pred, y_true))
    accuracy = correct / len(y_true)

    return accuracy, len(y_true), skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Majority Voting (MV) baseline on crowdsourcing datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["trec2011", "UC"],
        help="Dataset directories located under the data root.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory that contains the dataset folders.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        try:
            accuracy, evaluated, skipped = evaluate_dataset(dataset, args.data_root)
        except Exception as exc:
            print(f"[{dataset}] Evaluation failed: {exc}")
            continue

        print(
            f"[{dataset}] MV accuracy: {accuracy:.4f} "
            f"(evaluated objects: {evaluated}, missing votes: {skipped})"
        )


if __name__ == "__main__":
    main()
