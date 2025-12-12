import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from collections import defaultdict
from typing import Dict, Optional
from data.process_demo import get_edge
from utils import update_feature, get_adj, update_reliability, dis_loss, eval, precision_recall_metrics
import Model
import tensorflow.python.ops.numpy_ops.np_config as np_config
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"tensorflow\.python\.ops\.numpy_ops\.np_dtypes"
)

## time with CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np_config.enable_numpy_behavior()


def sample_few_shot_labels(object_index,
                           source_index,
                           truth_set,
                           source_num,
                           ratio=0.02,
                           seed=42):
    truth_objects = truth_set['gt_index']
    truth_labels = truth_set['truths']
    if ratio <= 0 or len(truth_objects) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.zeros(source_num, dtype=np.float32)

    target = max(1, int(np.ceil(len(truth_objects) * ratio)))
    truth_map = {obj: label for obj, label in zip(truth_objects, truth_labels)}
    worker_to_objects = [set() for _ in range(source_num)]
    object_to_workers = defaultdict(set)
    for obj, worker in zip(object_index, source_index):
        worker_to_objects[worker].add(obj)
        object_to_workers[obj].add(worker)

    rng = np.random.default_rng(seed)
    worker_order = np.arange(source_num)
    rng.shuffle(worker_order)

    selected = []
    remaining = set(truth_objects.tolist())
    worker_mask = np.zeros(source_num, dtype=np.float32)

    while len(selected) < target:
        progress = False
        for worker in worker_order:
            if len(selected) >= target:
                break
            eligible = [obj for obj in worker_to_objects[worker] if obj in remaining and obj in truth_map]
            if not eligible:
                continue
            obj = rng.choice(eligible)
            selected.append(obj)
            remaining.discard(obj)
            worker_mask[worker] = 1.0
            progress = True
        if not progress:
            break

    if len(selected) < target and remaining:
        remaining_list = list(remaining)
        rng.shuffle(remaining_list)
        for obj in remaining_list:
            if len(selected) >= target:
                break
            selected.append(obj)
            remaining.discard(obj)
            workers = list(object_to_workers[obj])
            if workers:
                worker_mask[rng.choice(workers)] = 1.0

    for obj in selected:
        for worker in object_to_workers[obj]:
            worker_mask[worker] = 1.0

    selected = np.array(sorted(set(selected)), dtype=np.int32)
    labels = np.array([truth_map[obj] for obj in selected], dtype=np.int32)

    return selected, labels, worker_mask


def build_eval_truth_set(truth_set, excluded_objects):
    if excluded_objects.size == 0:
        return truth_set
    mask = ~np.isin(truth_set['gt_index'], excluded_objects)
    if not np.any(mask):
        warnings.warn('Few-shot objects cover entire truth set; using full truth for evaluation.')
        return truth_set
    return {
        'truths': truth_set['truths'][mask],
        'gt_index': truth_set['gt_index'][mask]
    }


def apply_supervised_object_features(features,
                                     object_num,
                                     class_num,
                                     few_shot_indices_tf,
                                     few_shot_labels_tf,
                                     strength):
    if few_shot_indices_tf is None or strength <= 0.0:
        return features
    strength = tf.convert_to_tensor(strength, dtype=features.dtype)
    object_features = features[:object_num]
    supervised_features = tf.gather(object_features, few_shot_indices_tf)
    truth_one_hot = tf.one_hot(few_shot_labels_tf, depth=class_num, dtype=features.dtype)
    updated = supervised_features + strength * (truth_one_hot - supervised_features)
    object_features = tf.tensor_scatter_nd_update(
        object_features,
        tf.expand_dims(few_shot_indices_tf, axis=1),
        updated
    )
    object_features = tf.nn.softmax(object_features, axis=-1)
    return tf.concat([object_features, features[object_num:]], axis=0)


def run_tiremge(
    answer_path: str,
    truth_path: str,
    dataset_name: Optional[str] = None,
    metrics_dir: Optional[str] = None,
    few_shot_ratio: float = 0.5,
    supervised_loss_weight: float = 1.0,
    supervised_reliability_boost: float = 0.0,
    object_supervision_strength: float = 0.0,
    few_shot_seed: int = 42,
    log_data_sample_interval: int = 10,
    steps: int = 200,
    learning_rate: float = 1e-2,
    save_outputs: bool = True,
) -> Dict[str, float]:
    answer_path = str(answer_path)
    truth_path = str(truth_path)
    dataset_label = dataset_name or Path(answer_path).parent.name
    graph, object_index, source_index, truth_set = get_edge(answer_path=answer_path, truth_path=truth_path)
    object_source_pair = graph['object_source_pair']
    node_num = np.max(object_source_pair) + 1
    object_num = np.max(object_source_pair[0]) + 1
    source_num = node_num - object_num
    class_num = int(np.max(truth_set['truths']) + 1)
    claims = tf.one_hot(indices=graph['claims'], depth=class_num)
    truth_lookup = {int(obj): int(label) for obj, label in zip(truth_set['gt_index'], truth_set['truths'])}

    few_shot_indices, few_shot_labels, worker_supervision_mask = sample_few_shot_labels(
        object_index,
        source_index,
        truth_set,
        source_num,
        ratio=few_shot_ratio,
        seed=few_shot_seed,
    )
    few_shot_count = few_shot_indices.shape[0]
    eval_truth_set = build_eval_truth_set(truth_set, few_shot_indices)

    print("few_shot_labels: ", few_shot_labels)
    print("worker_supervision_mask:", worker_supervision_mask)

    if few_shot_count > 0:
        few_shot_indices_tf = tf.constant(few_shot_indices, dtype=tf.int32)
        few_shot_labels_tf = tf.constant(few_shot_labels, dtype=tf.int32)
    else:
        few_shot_indices_tf = None
        few_shot_labels_tf = None
    worker_supervision_mask_tf = tf.constant(worker_supervision_mask, dtype=tf.float32)

    worker_accuracy_history = []

    if few_shot_count > 0:
        covered_workers = int(np.sum(worker_supervision_mask))
        coverage = few_shot_count / max(len(truth_set['gt_index']), 1)
        print(
            f'Few-shot supervision enabled: using {few_shot_count} labels '
            f'({coverage:.2%}) covering {covered_workers} workers.'
        )
    else:
        print('Few-shot supervision disabled: no labels selected.')

    adj1, adj2 = get_adj(object_source_pair, node_num)
    edge_index1 = adj1.astype(np.int32)
    edge_index2 = adj2.astype(np.int32)

    model = Model.TiReMGE(node_num=node_num, source_num=source_num, class_num=class_num)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    tf.train.Checkpoint(model=model, optimizer=optimizer)

    reliability = np.ones(shape=(source_num)) / source_num
    x = update_feature([object_index, source_index, claims], reliability, object_num, source_num)
    x = apply_supervised_object_features(
        x,
        object_num,
        class_num,
        few_shot_indices_tf,
        few_shot_labels_tf,
        object_supervision_strength,
    )

    best_acc = 0.0
    best_embedding = None
    best_hidden = None
    hidden = None

    for step in range(steps):
        with tf.GradientTape() as tape:
            embedding, hidden = model([x, edge_index1, edge_index2], return_hidden=True)

            reliability = update_reliability(embedding, [object_index, source_index, claims], source_num, reliability)
            loss1 = dis_loss(
                embedding,
                [object_index, source_index, claims],
                reliability,
                source_num,
                worker_supervision_mask=worker_supervision_mask_tf,
                supervision_boost=supervised_reliability_boost,
            )
            loss = loss1

            if few_shot_indices_tf is not None:
                labeled_logits = tf.gather(embedding, few_shot_indices_tf)
                supervision_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=few_shot_labels_tf,
                        logits=labeled_logits,
                    )
                )
                loss += supervised_loss_weight * supervision_loss

            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
        x = update_feature([object_index, source_index, claims], reliability, object_num, source_num)
        x = apply_supervised_object_features(
            x,
            object_num,
            class_num,
            few_shot_indices_tf,
            few_shot_labels_tf,
            object_supervision_strength,
        )

        if eval_truth_set['gt_index'].size > 0:
            acc = eval(embedding, eval_truth_set, class_num)
        else:
            acc = np.nan
        if not np.isnan(acc) and acc > best_acc:
            best_acc = acc
            best_embedding = tf.identity(embedding)
            best_hidden = tf.identity(hidden)
        print(f"step = {step}\tloss = {loss}\tbest_accuracy = {best_acc}\taccuracy = {acc}")
        if step % log_data_sample_interval == 0 and truth_set['gt_index'].size > 0:
            sample_pool = truth_set['gt_index']
            sample_count = min(5, sample_pool.size)
            sample_objects = np.random.choice(sample_pool, size=sample_count, replace=False)
            logits_sample = tf.gather(embedding, sample_objects)
            probs = tf.nn.softmax(logits_sample, axis=-1).numpy()
            preds = np.argmax(probs, axis=-1)
            log_entries = []
            for obj, pred, prob_vec in zip(sample_objects, preds, probs):
                obj = int(obj)
                truth_value = truth_lookup.get(obj)
                log_entries.append(
                    {
                        'object_index': obj,
                        'model_inference': {
                            'predicted_label': int(pred),
                            'confidence': float(prob_vec[pred]),
                        },
                        'truth': None if truth_value is None else truth_value,
                    }
                )
            print(f"Prediction sample at step {step}: {log_entries}")
        if truth_set['gt_index'].size > 0:
            supervised_mask = np.isin(object_index, truth_set['gt_index'])
            if np.any(supervised_mask):
                supervised_obj_idx = object_index[supervised_mask]
                supervised_workers = source_index[supervised_mask]
                supervised_answers = graph['claims'][supervised_mask]
                obj_to_truth = {obj: truth_lookup.get(int(obj)) for obj in np.unique(supervised_obj_idx)}
                worker_correct = np.zeros(source_num, dtype=np.float32)
                worker_total = np.zeros(source_num, dtype=np.float32)
                for obj, worker, answer in zip(supervised_obj_idx, supervised_workers, supervised_answers):
                    truth_value = obj_to_truth.get(int(obj))
                    if truth_value is None:
                        continue
                    worker_total[worker] += 1.0
                    if int(answer) == int(truth_value):
                        worker_correct[worker] += 1.0
                with np.errstate(divide='ignore', invalid='ignore'):
                    worker_accuracy = np.divide(
                        worker_correct,
                        worker_total,
                        out=np.zeros_like(worker_correct),
                        where=worker_total > 0,
                    )
                worker_accuracy_history.append(worker_accuracy)

    if best_embedding is None:
        best_embedding = embedding
    if best_hidden is None:
        best_hidden = hidden

    pr_metrics = None
    if eval_truth_set['gt_index'].size > 0:
        pr_metrics = precision_recall_metrics(best_embedding, eval_truth_set, class_num)

    metrics_path = None
    if save_outputs:
        metrics_target = Path(metrics_dir) if metrics_dir else Path('metrics') / dataset_label
        metrics_target.mkdir(parents=True, exist_ok=True)

        object_embeddings = best_hidden[:object_num].numpy()
        np.save(metrics_target / 'object_embeddings.npy', object_embeddings)
        object_logits = best_embedding[:object_num].numpy()
        np.save(metrics_target / 'object_logits.npy', object_logits)
        np.save(metrics_target / 'object_ids.npy', np.arange(object_num))
        np.save(metrics_target / 'ground_truth_indices.npy', eval_truth_set['gt_index'])
        np.save(metrics_target / 'labels.npy', eval_truth_set['truths'])

        if few_shot_count > 0:
            np.save(metrics_target / 'supervised_object_indices.npy', few_shot_indices)
            np.save(metrics_target / 'supervised_object_labels.npy', few_shot_labels)

        if worker_accuracy_history:
            worker_accuracy_array = np.stack(worker_accuracy_history, axis=0)
            np.save(metrics_target / 'worker_accuracy_history.npy', worker_accuracy_array)
            np.savetxt(metrics_target / 'worker_accuracy_history.txt', worker_accuracy_array, fmt='%.6f')

        if pr_metrics is not None:
            np.save(metrics_target / 'probabilities.npy', pr_metrics['probabilities'])
            if class_num == 2:
                np.save(metrics_target / 'precision.npy', pr_metrics['precision'])
                np.save(metrics_target / 'recall.npy', pr_metrics['recall'])
                np.save(metrics_target / 'thresholds.npy', pr_metrics['thresholds'])
                summary = {
                    'average_precision': float(pr_metrics['average_precision']),
                    'best_accuracy': float(best_acc),
                }
            else:
                precision_kwargs = {f'class_{cls}': arr for cls, arr in pr_metrics['precision'].items()}
                recall_kwargs = {f'class_{cls}': arr for cls, arr in pr_metrics['recall'].items()}
                thresholds_kwargs = {f'class_{cls}': arr for cls, arr in pr_metrics['thresholds'].items()}
                np.savez(metrics_target / 'precision.npz', **precision_kwargs)
                np.savez(metrics_target / 'recall.npz', **recall_kwargs)
                np.savez(metrics_target / 'thresholds.npz', **thresholds_kwargs)
                summary = {
                    'average_precision': {
                        f'class_{cls}': float(score) for cls, score in pr_metrics['average_precision'].items()
                    },
                    'best_accuracy': float(best_acc),
                }
        else:
            summary = {
                'average_precision': None,
                'best_accuracy': float(best_acc),
            }
            print('No evaluation truth available; skipped precision-recall export.')

        with (metrics_target / 'summary.json').open('w') as f:
            json.dump(summary, f, indent=2)

        print(f'Precision-recall data saved to {metrics_target}')
        metrics_path = str(metrics_target)

    return {
        'best_accuracy': float(best_acc),
        'metrics_dir': metrics_path,
        'few_shot_count': int(few_shot_count),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TiReMGE on a dataset stored under data/<name>.")
    parser.add_argument('--dataset', default='trec2011', help='Dataset name located under the data root directory.')
    parser.add_argument('--data-root', default='data', help='Root directory that contains <dataset>/answer.csv and truth.csv.')
    parser.add_argument('--metrics-root', default='metrics', help='Directory to save metrics outputs (per dataset).')
    parser.add_argument('--few-shot-ratio', type=float, default=0.5, help='Fraction of truths used for supervision.')
    parser.add_argument('--few-shot-seed', type=int, default=42, help='Random seed for few-shot sampling.')
    parser.add_argument('--supervised-loss-weight', type=float, default=1.0, help='Weight for few-shot supervision loss.')
    parser.add_argument('--supervised-reliability-boost', type=float, default=0.0, help='Boost applied to supervised workers.')
    parser.add_argument('--object-supervision-strength', type=float, default=0.0, help='Strength when fusing supervised object features.')
    parser.add_argument('--log-interval', type=int, default=10, help='Steps between prediction sample logs.')
    parser.add_argument('--steps', type=int, default=200, help='Number of training steps to run TiReMGE.')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate for the TiReMGE optimizer.')
    parser.add_argument('--no-save', action='store_true', help='Skip writing metrics/embeddings to disk.')
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    dataset_dir = Path(args.data_root) / args.dataset
    answer_path = dataset_dir / 'answer.csv'
    truth_path = dataset_dir / 'truth.csv'
    if not answer_path.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")
    metrics_dir = None if args.no_save else str(Path(args.metrics_root) / args.dataset)
    result = run_tiremge(
        answer_path=str(answer_path),
        truth_path=str(truth_path),
        dataset_name=args.dataset,
        metrics_dir=metrics_dir,
        few_shot_ratio=args.few_shot_ratio,
        supervised_loss_weight=args.supervised_loss_weight,
        supervised_reliability_boost=args.supervised_reliability_boost,
        object_supervision_strength=args.object_supervision_strength,
        few_shot_seed=args.few_shot_seed,
        log_data_sample_interval=args.log_interval,
        steps=args.steps,
        learning_rate=args.learning_rate,
        save_outputs=not args.no_save,
    )
    print(f"Best accuracy: {result['best_accuracy']:.4f}")


if __name__ == '__main__':
    main()
