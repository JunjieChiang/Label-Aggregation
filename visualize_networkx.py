#!/usr/bin/env python3
"""Visualize TiReMGE crowdsourcing datasets with NetworkX."""
import argparse
import csv
import os
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx

Edge = Tuple[str, str, int]


def load_edges(dataset: str, data_root: str) -> List[Edge]:
    answer_path = os.path.join(data_root, dataset, 'answer.csv')
    edges: List[Edge] = []
    with open(answer_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            obj, src = row[:2]
            claim = int(row[2]) if len(row) > 2 and row[2] != '' else 0
            edges.append((obj, src, claim))
    return edges


def select_subgraph(edges: Sequence[Edge], max_sources: int, max_objects: int) -> List[Edge]:
    source_counts = Counter(src for _, src, _ in edges)
    if max_sources and len(source_counts) > max_sources:
        selected_sources = {src for src, _ in source_counts.most_common(max_sources)}
    else:
        selected_sources = set(source_counts)

    filtered_edges = [edge for edge in edges if edge[1] in selected_sources]

    object_counts = Counter(obj for obj, _, _ in filtered_edges)
    if max_objects and len(object_counts) > max_objects:
        selected_objects = {obj for obj, _ in object_counts.most_common(max_objects)}
    else:
        selected_objects = set(object_counts)

    return [edge for edge in filtered_edges if edge[0] in selected_objects]


def build_graph(edges: Iterable[Edge]) -> nx.Graph:
    G = nx.Graph()
    for obj, src, claim in edges:
        object_node = f'obj::{obj}'
        source_node = f'src::{src}'
        G.add_node(object_node, bipartite=0, label=obj, kind='object')
        G.add_node(source_node, bipartite=1, label=src, kind='source')
        G.add_edge(object_node, source_node, claim=claim)
    return G


def compute_layout(G: nx.Graph, object_nodes: Sequence[str], layout: str, seed: Optional[int]) -> Dict[str, Tuple[float, float]]:
    if layout == 'spring':
        return nx.spring_layout(G, seed=seed)
    if layout == 'kamada':
        return nx.kamada_kawai_layout(G)
    return nx.bipartite_layout(G, object_nodes, scale=1, align='vertical')


def plot_graph(
    dataset: str,
    G: nx.Graph,
    edge_claims: Sequence[int],
    output_dir: str,
    dpi: int,
    layout: str,
    seed: Optional[int],
    color_claims: bool,
    show_legend: bool,
) -> None:
    object_nodes = [n for n, data in G.nodes(data=True) if data['kind'] == 'object']
    source_nodes = [n for n, data in G.nodes(data=True) if data['kind'] == 'source']

    if not object_nodes or not source_nodes:
        raise ValueError('Graph is empty after sampling; try increasing max nodes.')

    pos = compute_layout(G, object_nodes, layout, seed)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    task_color = '#d62728'
    worker_color = '#1f77b4'

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=object_nodes,
        node_color=task_color,
        node_size=18,
        alpha=0.85,
        linewidths=0,
        ax=ax,
        label='Tasks',
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=source_nodes,
        node_color=worker_color,
        node_size=32,
        alpha=0.9,
        linewidths=0,
        ax=ax,
        label='Workers',
    )

    edge_colors: Sequence = '#2f2f2f'
    legend_handles: List[plt.Line2D] = []

    if color_claims and len(set(edge_claims)) > 1:
        unique_claims = sorted(set(edge_claims))
        cmap = plt.get_cmap('tab10', len(unique_claims))
        color_map: Dict[int, Tuple[float, float, float, float]] = {claim: cmap(i) for i, claim in enumerate(unique_claims)}
        edge_colors = [color_map[claim] for claim in edge_claims]
        legend_handles = [
            plt.Line2D([0], [0], color=color_map[claim], linewidth=0.8, label=f'Claim {claim}')
            for claim in unique_claims
        ]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.22, width=0.6, ax=ax)

    title = f"{dataset} task-worker subgraph\nNodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}"
    ax.set_title(title, fontsize=13)
    ax.axis('off')

    if show_legend:
        legend_handles.extend([
            plt.Line2D([0], [0], marker='o', color='w', label='Task', markerfacecolor=task_color, markersize=6),
            plt.Line2D([0], [0], marker='o', color='w', label='Worker', markerfacecolor=worker_color, markersize=7),
        ])
        ax.legend(handles=legend_handles, loc='upper right', frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset}_task_worker.png')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved visualization for {dataset} to {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualize crowdsourcing datasets as task-worker graphs using NetworkX.'
    )
    parser.add_argument('--datasets', nargs='+', default=['trec2011', 'UC'], help='Dataset names located under the data directory.')
    parser.add_argument('--data-root', default='data', help='Root directory containing dataset folders.')
    parser.add_argument('--output-dir', default='figures', help='Directory to save generated visualizations.')
    parser.add_argument('--max-sources', type=int, default=30, help='Maximum number of workers to include in the subgraph.')
    parser.add_argument('--max-objects', type=int, default=400, help='Maximum number of tasks to include in the subgraph.')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved figures.')
    parser.add_argument('--layout', choices=['spring', 'bipartite', 'kamada'], default='spring', help='Layout algorithm to position nodes.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for force-directed layouts.')
    parser.add_argument('--color-claims', action='store_true', help='Color edges by claim label instead of monochrome.')
    parser.add_argument('--hide-legend', action='store_true', help='Hide legend for a cleaner, publication-style graphic.')

    args = parser.parse_args()

    for dataset in args.datasets:
        edges = load_edges(dataset, args.data_root)
        if not edges:
            print(f'Skipping {dataset}: no edges found.')
            continue
        sub_edges = select_subgraph(edges, args.max_sources, args.max_objects)
        if not sub_edges:
            print(f'Skipping {dataset}: sampling produced empty subgraph.')
            continue
        G = build_graph(sub_edges)
        plot_graph(
            dataset,
            G,
            [claim for _, _, claim in sub_edges],
            args.output_dir,
            args.dpi,
            args.layout,
            args.seed,
            args.color_claims,
            not args.hide_legend,
        )


if __name__ == '__main__':
    main()
