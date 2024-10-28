import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx
import random


sys.stdout.flush()

num_workers = 10
#sample_count = 3

# 将子目录添加到 sys.path
current_dir = os.getcwd()

#sys.path.append(os.path.join(current_dir, 'EffectiveResistanceSampling'))
#from EffectiveResistanceSampling.Network import *

sys.path.append(os.path.join(current_dir, 'utilities'))
from utilities.tools import *

def remove_edges_with_retry(graph, percentage):
    num_edges = graph.number_of_edges()
    num_remove = int(percentage * num_edges)

    while True:
        print("trying...")
        # 创建副本以避免对原始图进行操作
        temp_graph = graph.copy()

        # 随机选择要删除的边
        edges = list(temp_graph.edges())
        edges_to_remove = random.sample(edges, num_remove)

        # 删除选中的边
        temp_graph.remove_edges_from(edges_to_remove)

        # 检查图是否连通
        if nx.is_connected(temp_graph):
            return temp_graph  # 返回连通的图

def random_graph_mu(mu, graph_type, percent):# output_dir=f'graph_random_{percent}'):
    """Process a specific mixing parameter (mu) to get graphs with some edges randomly deleted."""
    output_dir = f'graph_random_{percent}
    graphs = load_graph_only(mu, graph_type, "original")
    sample = len(graphs)

    graph_random = []

    for i in range(sample):
        G = graphs[i]

        G_random = remove_edges_with_retry(G, 1-percent)

        graph_random.append(G_random)

    # Save all graphs and memberships into a single file
    combined_data = {
        'graphs': graph_random,
    }

    os.makedirs(output_dir, exist_ok=True)

    mu_str = f"{mu:.2f}"
    file_path = os.path.join(output_dir, f'{graph_type}_graph_random_mu{mu_str}.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(combined_data, file)

    print(f'Saved all graphs for mu={mu} to {file_path}')


# Compute in parallel using ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete some edges randomly from graph.")
    parser.add_argument('--graph_type', type=str, choices=['ppm', 'lfr'], default='ppm',
                        help="Random graph type (ppm or lfr).")
    parser.add_argument('--percent', type=float, help="Percentage of edges to keep.")
    parser.add_argument('--start_step', type=float, default=0.05, help="start_step.")

    args = parser.parse_args()
    graph_type = args.graph_type
    start_step = args.start_step
    percent = args.percent

    if graph_type == "ppm":
        end_step = 0.9
    elif graph_type == "lfr":
        end_step = 0.5
    step_size = 0.05
    MU = np.around(np.arange(start_step, end_step + 0.01, step_size), decimals=2)


    def process_mu(mu):
        return random_graph_mu(mu, graph_type)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_mu, MU))


