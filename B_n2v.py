import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx

embedding_dim = 16

sys.stdout.flush()

# 将子目录添加到 sys.path
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir,'EffectiveResistanceSampling'))
from EffectiveResistanceSampling.Network import *

sys.path.append(os.path.join(current_dir,'utilities'))
from utilities.tools import *


from node2vec import Node2Vec

    

walk_length=40
window=10 
num_walks=80
workers=16
min_count=1 
batch_words=6192


def community_detection(mu, graph_type, delete_type, percent):
    """Process a specific mixing parameter (mu) to do community detection."""
    original_graphs, memberships  = load_graph(mu, graph_type, "original", percent)
    graphs = load_graph_only(mu, graph_type, delete_type, percent)
    nodes = [str(i) for i in range(graphs[0].number_of_nodes())]
    sample  = len(graphs)
    detected_euclid_memberships = []
    detected_cosine_memberships = []
    raw_qf_mu = np.zeros((sample))

    for i in range(sample):
        G = graphs[i]
        G_orig = original_graphs[i]
        intrinsic_membership = memberships[i]
        K = len(np.unique(intrinsic_membership))

        
        A = nx.to_numpy_array(G_orig, nodelist=G_orig.nodes(), weight='weight', dtype=np.float64)

        node2vec_original = Node2Vec(G, dimensions=embedding_dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
        node2vec_fit_original = node2vec_original.fit(window=window, min_count=min_count, batch_words=batch_words) 
        embedding = np.array([node2vec_fit_original.wv[node] for node in nodes])

        detected_euclid_memberships.append(get_euclid_membership(K, embedding))
        detected_cosine_memberships.append(get_cosine_membership(K, embedding))

        # Potential optimization of the quadratic form calculation using CuPy
        quadratic_form_original = 0
        for k in range(embedding.shape[1]):
            vk = embedding[:, k]
            diff = vk[:, np.newaxis] - vk[np.newaxis, :]  # 创建一个差的矩阵
            quadr = A * (diff ** 2)  # 使用广播计算二次项
            quadratic_form_original += np.sum(quadr)  # 累加

        raw_qf_mu[i] = quadratic_form_original
        
        print(mu, i)

    # 创建 community_detection 目录（如果不存在）
    os.makedirs(f'communitydetection_{delete_type}', exist_ok=True)
    if delete_type == "original":
        cd_output_dir = f'communitydetection_{delete_type}'
    else:
        cd_output_dir = f'communitydetection_{delete_type}_{percent}'
    os.makedirs(output_dir, exist_ok=True)

    # Save memberships for this specific mu
    mu_str = f"{mu:.2f}"
    raw_euclid_path = f'{cd_output_dir}/{graph_type}_{delete_type}_n2v_euclid_mu{mu_str}.pkl'
    with open(raw_euclid_path, 'wb') as file:
        pickle.dump(detected_euclid_memberships, file)
    print(f"Euclid membership for mu={mu_str} saved to {raw_euclid_path}")

    raw_cosine_path = f'{cd_output_dir}/{graph_type}_{delete_type}_n2v_cosine_mu{mu_str}.pkl'
    with open(raw_cosine_path, 'wb') as file:
        pickle.dump(detected_cosine_memberships, file)
    print(f"Cosine membership for mu={mu_str} saved to {raw_cosine_path}")

    if delete_type == "original":
        result_output_dir = f'results_{delete_type}'
    else:
        result_output_dir = f'results_{delete_type}_{percent}'
    os.makedirs(result_output_dir, exist_ok=True)
    raw_qf_path = f'{result_output_dir}/{graph_type}_{delete_type}_n2v_raw_qf_mu{mu_str}.pkl'
    with open(raw_qf_path, 'wb') as file:
        pickle.dump(raw_qf_mu, file)
    print(f"RAW_QF for mu={mu_str} saved to {raw_qf_path}")

def main():
    parser = argparse.ArgumentParser(description="Community detection on networks with different mu.")
    parser.add_argument('--graph_type', type=str, choices=['ppm', 'lfr'], default='ppm', help="Random graph type (ppm or lfr)")
    parser.add_argument('--start_step', type=float, default=0.05, help="start_step")
    parser.add_argument('--delete_type', type=str, choices=['original', 'sparse', 'random'], help="Ways to delete edges (original, sparse, or random)")
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