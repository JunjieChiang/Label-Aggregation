#!/usr/bin/env python3
"""Project TiReMGE object embeddings into 2D for visualization."""
import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE


def load_embeddings(metrics_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    embedding_path = os.path.join(metrics_dir, 'object_embeddings.npy')
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f'Missing embeddings at {embedding_path}. Run demo.py first to generate metrics.')
    embeddings = np.load(embedding_path)

    labels = None
    labels_path = os.path.join(metrics_dir, 'labels.npy')
    gt_idx_path = os.path.join(metrics_dir, 'ground_truth_indices.npy')
    if os.path.exists(labels_path) and os.path.exists(gt_idx_path):
        truth_labels = np.load(labels_path)
        gt_indices = np.load(gt_idx_path)
        labels = np.full(embeddings.shape[0], -1, dtype=int)
        labels[gt_indices] = truth_labels
    return embeddings, labels


def reduce_dim(embeddings: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(embeddings)
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=seed, init='pca', learning_rate='auto', perplexity=50)
        return reducer.fit_transform(embeddings)
    if method == 'umap':
        reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
        return reducer.fit_transform(embeddings)
    raise ValueError(f'Unknown reduction method: {method}')


def build_colors(labels: np.ndarray) -> Tuple[np.ndarray, list, list]:
    if labels is None:
        return np.array(['#4a1486']), [], []

    valid = labels >= 0
    unique = sorted(set(labels[valid]))
    if not unique:
        return np.array(['#4a1486']), [], []

    base_colors = ['#4a1486', '#fdd835', '#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    cmap = ListedColormap(base_colors[:max(len(unique), 2)])
    color_lookup = {cls: mcolors.to_hex(cmap(i % len(base_colors))) for i, cls in enumerate(unique)}

    colors = np.array([color_lookup.get(int(lbl), '#bdbdbd') for lbl in labels])
    legend_elements = [
        plt.Line2D([0], [0], marker='o', linestyle='', color=color_lookup[cls], label=f'Class {cls}')
        for cls in unique
    ]
    masked_indices = np.where(labels < 0)[0]
    return colors, legend_elements, masked_indices.tolist()


def plot_embeddings(points: np.ndarray, colors: np.ndarray, masked: list, title: str, output: str, legend_elements: list, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

    if colors.size == 1:
        ax.scatter(points[:, 0], points[:, 1], s=16, c=colors[0], alpha=0.75, edgecolors='none')
    else:
        known_mask = np.array([i not in masked for i in range(points.shape[0])])
        if known_mask.any():
            ax.scatter(points[known_mask, 0], points[known_mask, 1], s=18, c=colors[known_mask], alpha=0.85, edgecolors='none')
        if masked:
            ax.scatter(points[masked, 0], points[masked, 1], s=14, c='#bdbdbd', alpha=0.4, edgecolors='none', label='Unlabelled')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if legend_elements:
        handles = legend_elements.copy()
        if masked:
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color='#bdbdbd', label='Unlabelled'))
        ax.legend(handles=handles, frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize TiReMGE task embeddings in 2D.')
    parser.add_argument('--dataset', default='trec2011', help='Dataset name (metrics/<dataset>/).')
    parser.add_argument('--metrics-root', default='metrics', help='Directory where demo.py saved metrics.')
    parser.add_argument('--method', choices=['tsne', 'pca', 'umap'], default='tsne', help='Dimensionality reduction algorithm.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for stochastic reducers.')
    parser.add_argument('--output', help='Output image path; defaults to metrics/<dataset>/task_embeddings_<method>.png')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively.')

    args = parser.parse_args()

    metrics_dir = os.path.join(args.metrics_root, args.dataset)
    embeddings, labels = load_embeddings(metrics_dir)
    points = reduce_dim(embeddings, args.method, args.seed)

    colors, legend_elements, masked = build_colors(labels)
    output_path = args.output or os.path.join(metrics_dir, f'task_embeddings_{args.method}.png')
    title = f'{args.dataset} (TiReMGE)'
    plot_embeddings(points, colors, masked, title, output_path, legend_elements, args.show)
    print(f'Saved embedding visualization to {output_path}')


if __name__ == '__main__':
    main()
