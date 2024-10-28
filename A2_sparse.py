import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx

sys.stdout.flush()

num_workers = 10
#sample_count = 3

# 将子目录添加到 sys.path
current_dir = os.getcwd()
sys.path.append(os.path.join(os.path.join(current_dir), 'EffectiveResistanceSampling'))
from EffectiveResistanceSampling.Network import *

sys.path.append(os.path.join(os.path.join(current_dir), 'utilities'))
from utilities.tools import *


def sparse_graph_mu(mu, graph_type, percent, epsilon=0.1):#, output_dir='graph_sparse'):
    output_dir = f'graph_sparse_{percent}'
    """Process a specific mixing parameter (mu) to get sparsed graphs."""
    graphs = load_graph_only(mu, graph_type, "original", percent)
    sample = len(graphs)
    if percent == 0.9:
        q_values = {"lfr": 46000, "ppm": 30000}
    elif percent == 0.6:
        q_values = {"lfr": 18000, "ppm": 12000} 
    graph_sparse = []

    for i in range(sample):
        G = graphs[i]

        edge_list = list(G.edges())
        edge_list = np.array(edge_list)

        edge_weights = nx.get_edge_attributes(G, 'weight')
        edge_weights = np.array(edge_weights)
        edge_weights = np.array([edge_weights[edge] if edge in edge_weights else 1 for edge in edge_list])

        Gn = Network(edge_list, edge_weights)
        Effective_R = Gn.effR(epsilon, 'spl')

        while True:
            Gn_Sparse = Gn.spl(q_values[graph_type], Effective_R, seed=2024)  # 第一个参数是 q 是边有放回抽样的数量
            G_sparse = to_networkx(Gn_Sparse)
            if nx.is_connected(G_sparse):
                break

        print(i)
        graph_sparse.append(G_sparse)

    # Save all graphs and memberships into a single file
    combined_data = {
        'graphs': graph_sparse,
    }

    os.makedirs(output_dir, exist_ok=True)

    mu_str = f"{mu:.2f}"
    file_path = os.path.join(output_dir, f'{graph_type}_graph_sparse_mu{mu_str}.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(combined_data, file)

    print(f'Saved all sparse graphs for mu={mu} to {file_path}')


# Compute in parallel using ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get a sparse version of the graph.")
    parser.add_argument('--graph_type', type=str, choices=['ppm', 'lfr'], default='ppm',
                        help="Random graph type (ppm or lfr)")
    parser.add_argument('--percent', type=float, help="Percentage of edges to keep.")
    parser.add_argument('--start_step', type=float, default=0.05, help="start_step")

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
        return sparse_graph_mu(mu, graph_type, percent)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_mu, MU))


