import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx
import cupy as cp


sys.stdout.flush()

# 将子目录添加到 sys.path
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir,'EffectiveResistanceSampling'))
from EffectiveResistanceSampling.Network import *

sys.path.append(os.path.join(current_dir,'utilities'))
from utilities.tools import *

def lap_cupy(graph, dim):
    """Compute Laplacian embedding of a graph using CuPy."""
    assert isinstance(graph, nx.Graph), "Input graph must be a NetworkX graph."
    assert isinstance(dim, int) and dim > 0, "Dimension must be a positive integer."
    assert dim < graph.number_of_nodes(), "Dimension must be less than the number of nodes."

    A = cp.asarray(nx.adjacency_matrix(graph, nodelist=graph.nodes(), weight='weight').toarray(), dtype=cp.float64)

    row_sums = cp.linalg.norm(A, ord=1, axis=1)
    P = A / row_sums.reshape(-1, 1)
    I_n = cp.eye(graph.number_of_nodes())
    w, v = cp.linalg.eigh(I_n - P)
    v = v[:, cp.argsort(w.real)]
    return v[:, 1:(dim + 1)].get().real


def community_detection(mu, graph_type, delete_type):
    """Process a specific mixing parameter (mu) to do community detection."""
    graphs, memberships  = load_graph(mu, graph_type, "original")
    sample  = len(graphs)
    detected_euclid_memberships = []
    detected_cosine_memberships = []
    raw_qf_mu = np.zeros((sample))

    for i in range(sample):
        G = graphs[i]
        intrinsic_membership = memberships[i]
        K = len(np.unique(intrinsic_membership))

        A = nx.to_numpy_array(G, nodelist=G.nodes(), weight='weight', dtype=np.float64)

        embedding = lap_cupy(G, K)

        detected_euclid_memberships.append(euclid_membership(K, embedding))
        detected_cosine_memberships.append(cosine_membership(K, embedding))

        quadratic_form = 0
        for k in range(embedding.shape[1]):
            vk = embedding[:, k]
            for s in range(A.shape[0]):
                for t in range(A.shape[1]):
                    quadr = A[s, t] * (vk[s] - vk[t]) ** 2
                    quadratic_form += quadr
        raw_qf_mu[i] = quadratic_form

        print(i)

        # 创建 community_detection 目录（如果不存在）
    os.makedirs(f'communitydetection_{delete_type}', exist_ok=True)

    # Save memberships for this specific mu
    mu_str = f"{mu:.2f}"
    output_dir = f'communitydetection_{delete_type}'
    raw_euclid_path = f'{output_dir}/{graph_type}_{delete_type}_lap_euclid_mu{mu_str}.pkl'
    with open(raw_euclid_path, 'wb') as file:
        pickle.dump(detected_euclid_memberships, file)
    print(f"Euclid membership for mu={mu_str} saved to {raw_euclid_path}")

    raw_cosine_path = f'{output_dir}/{graph_type}_{delete_type}_lap_cosine_mu{mu_str}.pkl'
    with open(raw_cosine_path, 'wb') as file:
        pickle.dump(detected_cosine_memberships, file)
    print(f"Cosine membership for mu={mu_str} saved to {raw_cosine_path}")

    os.makedirs(f'results_{delete_type}', exist_ok=True)
    raw_qf_path = f'results_{delete_type}/{graph_type}_{delete_type}_lap_raw_qf_mu{mu_str}.pkl'
    with open(raw_qf_path, 'wb') as file:
        pickle.dump(raw_qf_mu, file)
    print(f"RAW_QF for mu={mu_str} saved to {raw_qf_path}")


def main():
    parser = argparse.ArgumentParser(description="Community detection on networks with different mu.")
    parser.add_argument('--graph_type', type=str, choices=['ppm', 'lfr'], default='ppm',
                        help="Random graph type (ppm or lfr)")
    parser.add_argument('--start_step', type=float, default=0.05, help="start_step")
    parser.add_argument('--delete_type', type=str, choices=['original', 'sparse', 'random'],
                        help="Ways to delete edges (original, sparse, or random)")

    args = parser.parse_args()
    graph_type = args.graph_type
    start_step = args.start_step
    delete_type = args.delete_type

    if graph_type == "ppm":
        end_step = 0.9
    elif graph_type == "lfr":
        end_step = 0.5
    step_size = 0.05
    MU = np.around(np.arange(start_step, end_step + 0.01, step_size), decimals=2)

    print(MU)

    print("程序已经在运行啦！")

    for mu in MU:
        community_detection(mu, graph_type, delete_type)

    print("All tasks completed")


main()