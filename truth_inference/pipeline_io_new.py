from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _standardize_answer_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"object", "task", "task_id"}:
            col_map[col] = "object"
        elif lowered in {"worker", "annotator", "user"}:
            col_map[col] = "worker"
        elif lowered in {"response", "label", "answer"}:
            col_map[col] = "response"
    renamed = df.rename(columns=col_map)
    required = ["object", "worker", "response"]
    missing = set(required).difference(renamed.columns)
    if missing and len(renamed.columns) >= 3:
        fallback_map = {
            renamed.columns[0]: "object",
            renamed.columns[1]: "worker",
            renamed.columns[2]: "response",
        }
        renamed = renamed.rename(columns=fallback_map)
        missing = set(required).difference(renamed.columns)
    if missing:
        raise ValueError(f"Missing required columns in answer data: {sorted(missing)}")
    standardized = renamed[required].copy()
    standardized["object"] = standardized["object"].astype(str).str.strip()
    standardized["worker"] = standardized["worker"].astype(str).str.strip()
    standardized["response"] = standardized["response"].astype(str).str.strip()
    return standardized


def _standardize_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = str(col).strip().lower()
        if lowered in {"object", "task", "task_id"}:
            col_map[col] = "object"
        elif lowered in {"truth", "label", "gold"}:
            col_map[col] = "truth"
    renamed = df.rename(columns=col_map)
    required = ["object", "truth"]
    missing = set(required).difference(renamed.columns)
    if missing and len(renamed.columns) >= 2:
        fallback_map = {renamed.columns[0]: "object", renamed.columns[1]: "truth"}
        renamed = renamed.rename(columns=fallback_map)
        missing = set(required).difference(renamed.columns)
    if missing:
        raise ValueError(f"Missing required columns in truth data: {sorted(missing)}")
    standardized = renamed[required].copy()
    standardized["object"] = standardized["object"].astype(str).str.strip()
    standardized["truth"] = standardized["truth"].astype(str).str.strip()
    return standardized


def load_dataset(dataset_name: str, data_root: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_root) / dataset_name
    answer_path = base / "answer.csv"
    truth_path = base / "truth.csv"
    if not answer_path.exists():
        raise FileNotFoundError(f"answer.csv not found at {answer_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"truth.csv not found at {truth_path}")
    answers_raw = pd.read_csv(answer_path)
    truth_raw = pd.read_csv(truth_path)
    answers = _standardize_answer_columns(answers_raw)
    truth = _standardize_truth_columns(truth_raw)
    return answers, truth


def compute_accuracy(predictions: pd.Series, truth: pd.DataFrame) -> float:
    truth_series = truth.set_index("object")["truth"]
    common = predictions.index.intersection(truth_series.index)
    if len(common) == 0:
        return float("nan")
    aligned_pred = predictions.loc[common].astype(str)
    aligned_truth = truth_series.loc[common].astype(str)
    return (aligned_pred == aligned_truth).mean()
