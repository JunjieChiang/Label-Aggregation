#!/usr/bin/env python3
"""Plot precision-recall curve from saved TiReMGE metrics."""
import argparse
import json
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_root: str, dataset: str) -> Dict[str, Any]:
    metrics_dir = os.path.join(metrics_root, dataset)
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    metrics: Dict[str, Any] = {
        'labels': np.load(os.path.join(metrics_dir, 'labels.npy')),
        'probabilities': np.load(os.path.join(metrics_dir, 'probabilities.npy')),
        'summary': {}
    }

    summary_path = os.path.join(metrics_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            metrics['summary'] = json.load(f)

    precision_path = os.path.join(metrics_dir, 'precision.npy')
    if os.path.exists(precision_path):
        metrics['precision'] = np.load(precision_path)
        metrics['recall'] = np.load(os.path.join(metrics_dir, 'recall.npy'))
        metrics['thresholds'] = np.load(os.path.join(metrics_dir, 'thresholds.npy'))
    else:
        precision_npz = np.load(os.path.join(metrics_dir, 'precision.npz'))
        recall_npz = np.load(os.path.join(metrics_dir, 'recall.npz'))
        thresholds_npz = np.load(os.path.join(metrics_dir, 'thresholds.npz'))
        metrics['precision'] = {k: precision_npz[k] for k in precision_npz.files}
        metrics['recall'] = {k: recall_npz[k] for k in recall_npz.files}
        metrics['thresholds'] = {k: thresholds_npz[k] for k in thresholds_npz.files}

    return metrics


def plot_binary_pr(recall: np.ndarray, precision: np.ndarray, average_precision: float, output: str,
                    dpi: int, show: bool, add_f1: bool) -> None:
    plt.figure(figsize=(8, 8), dpi=dpi)
    plt.plot(recall, precision, linestyle='-', linewidth=2, color='#1f77b4', label='TiReMGE')

    if add_f1:
        recalls = np.linspace(0.001, 1, 1000)
        epsilon = 1e-7
        for f1_score in np.arange(0.1, 1.0, 0.1):
            denominator = (2 * recalls - f1_score)
            pr_values = np.where(denominator > 0, (recalls * f1_score) / denominator, 1 - epsilon)
            plt.plot(recalls, pr_values, linestyle='-', linewidth=1, color='#CCCCCC', alpha=0.7)
            plt.text(1.005, pr_values[-1], f'F1={f1_score:.1f}', fontsize=10, va='bottom')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    title = f'Precision-Recall Curve (AP = {average_precision:.3f})'
    plt.title(title, fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(False)

    if output:
        plt.savefig(output, bbox_inches='tight')
        print(f'Saved PR curve to {output}')

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot precision-recall curve from TiReMGE metrics output.')
    parser.add_argument('--dataset', default='trec2011', help='Dataset name used when running demo.py')
    parser.add_argument('--metrics-root', default='metrics', help='Directory where metrics are stored')
    parser.add_argument('--output', default='pr_curve.png', help='Path to save the PR curve image')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI for the saved image')
    parser.add_argument('--no-f1', action='store_true', help='Disable F1 guideline lines on the PR plot')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively')

    args = parser.parse_args()

    metrics = load_metrics(args.metrics_root, args.dataset)

    if isinstance(metrics['precision'], dict):
        raise ValueError('Multi-class PR plotting is not implemented in this script.')

    summary = metrics.get('summary', {})
    average_precision = summary.get('average_precision')
    if average_precision is None:
        average_precision = np.trapz(metrics['precision'], metrics['recall'])

    plot_binary_pr(
        recall=metrics['recall'],
        precision=metrics['precision'],
        average_precision=average_precision,
        output=args.output,
        dpi=args.dpi,
        show=args.show,
        add_f1=not args.no_f1
    )


if __name__ == '__main__':
    main()
