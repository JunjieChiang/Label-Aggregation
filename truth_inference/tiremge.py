from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

import Model
from utils import dis_loss, get_adj, update_feature, update_reliability

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
    object_indices = df["object"].map(obj_to_idx).to_numpy(dtype=np.int32)
    worker_indices = df["worker"].map(worker_to_idx).to_numpy(dtype=np.int32)
    label_indices = df["response"].map(label_to_idx).to_numpy(dtype=np.int32)
    return objects, workers, labels, object_indices, worker_indices, label_indices


def _progress_iter(num_steps: int, show_progress: bool, desc: str):
    iterable = range(num_steps)
    if not show_progress or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False)


def run_tiremge_prediction(
    answers: pd.DataFrame,
    *,
    max_steps: int,
    learning_rate: float,
    show_progress: bool,
    seed: int | None,
    desc: str = "TiReMGE training",
) -> Tuple[List[str], List[str], List[str], Dict[str, str], np.ndarray]:
    (
        objects,
        workers,
        labels,
        object_indices,
        worker_indices,
        label_indices,
    ) = _encode_entities(answers)
    predictions, probabilities = _train_tiremge(
        objects=objects,
        workers=workers,
        labels=labels,
        object_indices=object_indices,
        worker_indices=worker_indices,
        label_indices=label_indices,
        max_steps=max_steps,
        learning_rate=learning_rate,
        show_progress=show_progress,
        seed=seed,
        desc=desc,
    )
    return objects, workers, labels, predictions, probabilities


def _train_tiremge(
    *,
    objects: Sequence[str],
    workers: Sequence[str],
    labels: Sequence[str],
    object_indices: np.ndarray,
    worker_indices: np.ndarray,
    label_indices: np.ndarray,
    max_steps: int,
    learning_rate: float,
    show_progress: bool,
    seed: int | None,
    desc: str,
) -> Tuple[Dict[str, str], np.ndarray]:
    object_num = len(objects)
    source_num = len(workers)
    class_num = len(labels)
    if object_num == 0 or source_num == 0:
        raise ValueError("TiReMGE requires at least one object and one worker annotation.")
    if class_num == 0:
        raise ValueError("TiReMGE requires at least one unique label.")

    node_num = object_num + source_num
    object_source_pair = np.vstack([object_indices, worker_indices + object_num]).astype(np.int32)
    edge_index1, edge_index2 = get_adj(object_source_pair, node_num)
    edge_index1 = tf.cast(edge_index1, dtype=tf.int32)
    edge_index2 = tf.cast(edge_index2, dtype=tf.int32)
    object_index_tensor = tf.constant(object_indices, dtype=tf.int32)
    source_index_tensor = tf.constant(worker_indices, dtype=tf.int32)
    claims = tf.one_hot(label_indices, depth=class_num, dtype=tf.float32)

    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    model = Model.TiReMGE(node_num=node_num, source_num=source_num, class_num=class_num)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    triple = [object_index_tensor, source_index_tensor, claims]
    edge_index1_const = tf.identity(edge_index1)
    edge_index2_const = tf.identity(edge_index2)

    @tf.function(reduce_retracing=True)
    def train_step(features, reliability_vec):
        with tf.GradientTape() as tape:
            embedding_out = model([features, edge_index1_const, edge_index2_const], training=True)
            new_reliability = update_reliability(
                embedding_out,
                triple,
                source_num,
                reliability_vec,
            )
            loss = dis_loss(
                embedding_out,
                triple,
                new_reliability,
                source_num,
            )
        grads = tape.gradient(loss, model.trainable_variables)
        adjusted_grads = [
            tf.zeros_like(var) if grad is None else grad for grad, var in zip(grads, model.trainable_variables)
        ]
        optimizer.apply_gradients(zip(adjusted_grads, model.trainable_variables))
        updated_features = update_feature(
            triple,
            new_reliability,
            object_num,
            source_num,
        )
        return updated_features, new_reliability, embedding_out

    reliability = tf.fill([source_num], 1.0 / float(source_num))
    features = update_feature(
        triple,
        reliability,
        object_num,
        source_num,
    )

    embedding = None
    for _ in _progress_iter(max_steps, show_progress, desc):
        features, reliability, embedding = train_step(features, reliability)

    if embedding is None:
        embedding = model([features, edge_index1, edge_index2], training=False)

    object_embeddings = tf.gather(embedding, tf.range(object_num))
    probabilities = tf.nn.softmax(object_embeddings, axis=-1).numpy()
    pred_indices = tf.argmax(probabilities, axis=-1).numpy()
    predicted_labels = [labels[int(idx)] for idx in pred_indices]
    return dict(zip(objects, predicted_labels)), probabilities


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
