import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx

import igraph as ig
import leidenalg as la


sys.stdout.flush()

# 将子目录添加到 sys.path
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir,'EffectiveResistanceSampling'))
from EffectiveResistanceSampling.Network import *

sys.path.append(os.path.join(current_dir,'utilities'))
from utilities.tools import *


def community_detection(mu, graph_type, delete_type, percent):
    """Process a specific mixing parameter (mu) to do community detection."""
    graphs = load_graph_only(mu, graph_type, delete_type, percent)
    sample  = len(graphs)
    detected_euclid_memberships = []
    detected_cosine_memberships = []
    raw_qf_mu = np.zeros((sample))

    for i in range(sample):
        G = graphs[i]
        G_ig = ig.Graph.Adjacency((nx.to_numpy_array(G) > 0).tolist())

        partition = la.find_partition(G_ig, la.ModularityVertexPartition)

        detected_euclid_memberships.append(partition.membership)

        print(mu, i)

        # 创建 community_detection 目录（如果不存在）

    if delete_type == "original":
        cd_output_dir = f'leiden_communitydetection_{delete_type}'
    else:
        cd_output_dir = f'leiden_communitydetection_{delete_type}_{percent}'
    os.makedirs(cd_output_dir, exist_ok=True)

    # Save memberships for this specific mu
    mu_str = f"{mu:.2f}"
    raw_euclid_path = f'{cd_output_dir}/{graph_type}_{delete_type}_lap_euclid_mu{mu_str}.pkl'
    with open(raw_euclid_path, 'wb') as file:
        pickle.dump(detected_euclid_memberships, file)
    print(f"Euclid membership for mu={mu_str} saved to {raw_euclid_path}")


def main():
    parser = argparse.ArgumentParser(description="Community detection on networks with different mu.")
    parser.add_argument('--graph_type', type=str, choices=['ppm', 'lfr'], default='ppm',
                        help="Random graph type (ppm or lfr)")
    parser.add_argument('--start_step', type=float, default=0.05, help="start_step")
    parser.add_argument('--delete_type', type=str, choices=['original', 'sparse', 'random'],
                        help="Ways to delete edges (original, sparse, or random)")
    parser.add_argument('--percent', type=float, help="Percentage of edges to keep.")

    args = parser.parse_args()
    graph_type = args.graph_type
    start_step = args.start_step
    delete_type = args.delete_type
    percent = args.percent

    if graph_type == "ppm":
        end_step = 0.9
    elif graph_type == "lfr":
        end_step = 0.5
    step_size = 0.05
    MU = np.around(np.arange(start_step, end_step + 0.01, step_size), decimals=2)

    print(MU)

    print("程序已经在运行啦！")

    for mu in MU:
        community_detection(mu, graph_type, delete_type, percent)

    print("All tasks completed")


main()