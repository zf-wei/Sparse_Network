import argparse
import os
import sys
import pickle
import numpy as np
import networkx as nx

sys.stdout.flush()

# 将子目录添加到 sys.path
current_dir = os.getcwd()
sys.path.append(os.path.join(os.path.join(current_dir),'EffectiveResistanceSampling'))
from EffectiveResistanceSampling.Network import *

sys.path.append(os.path.join(os.path.join(current_dir),'utilities'))
from utilities.tools import *


from node2vec import Node2Vec

    
embedding_dim = 15
walk_length=40
num_walks=80
workers=16
window=10 
min_count=1 
batch_words=6192


def comprehensive_process(mu, graph_type, epsilon=0.1):
    """Process a specific mixing parameter (mu) to calculate scores and quadratic forms."""
    graphs, memberships = load_graph(mu, graph_type)
    nodes = [str(i) for i in range(graphs[0].number_of_nodes())]
    sample  = len(graphs)
    raw_score_mu = np.zeros((sample, 2, 4))
    raw_qf_mu = np.zeros((sample, 2))
    for i in range(sample):
        G = graphs[i]
        
        intrinsic_membership = memberships[i]
        K = len(np.unique(intrinsic_membership))

        edge_list = list(G.edges())
        edge_list = np.array(edge_list)

        edge_weights = nx.get_edge_attributes(G, 'weight')
        edge_weights = np.array(edge_weights)
        edge_weights = [edge_weights[edge] if edge in edge_weights else 1 for edge in edge_list]

        Gn = Network(edge_list, edge_weights)
        Effective_R = Gn.effR(epsilon, 'spl')

        while True:
            Gn_Sparse = Gn.spl(10000, Effective_R, seed=2024)
            G_sparse = to_networkx(Gn_Sparse)
            if nx.is_connected(G_sparse):
                break

        A = nx.to_numpy_array(G, nodelist=G.nodes(), weight='weight', dtype=np.float64) 
        node2vec_original = Node2Vec(G, dimensions=embedding_dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
        node2vec_fit_original = node2vec_original.fit(window=window, min_count=min_count, batch_words=batch_words) 
        embedding_original = np.array([node2vec_fit_original.wv[node] for node in nodes])

        node2vec_sparse = Node2Vec(G_sparse, dimensions=embedding_dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
        node2vec_fit_sparse = node2vec_sparse.fit(window=window, min_count=min_count, batch_words=batch_words)
        embedding_sparse = np.array([node2vec_fit_sparse.wv[node] for node in nodes])
        score_sparse = calculate_score(embedding_sparse, intrinsic_membership, K)
        raw_score_mu[i, 0] = score_sparse
        
        quadratic_form_sparse = 0
        for k in range(embedding_sparse.shape[1]):
            vk = embedding_sparse[:, k]
            for s in range(A.shape[0]):
                for t in range(A.shape[1]):
                    quadr = A[s, t] * (vk[s] - vk[t]) ** 2
                    quadratic_form_sparse += quadr
        raw_qf_mu[i,0] = quadratic_form_sparse
        
        score_original = calculate_score(embedding_original, intrinsic_membership, K)
        raw_score_mu[i, 1] = score_original

        quadratic_form_original = 0
        for k in range(embedding_original.shape[1]):
            vk = embedding_original[:, k]
            for s in range(A.shape[0]):
                for t in range(A.shape[1]):
                    quadr = A[s, t] * (vk[s] - vk[t]) ** 2
                    quadratic_form_original += quadr
            
        raw_qf_mu[i,1] = quadratic_form_original
        
        print(i)

    # 创建 results 目录（如果不存在）
    os.makedirs('results', exist_ok=True)

    # Save results for this specific mu
    mu_str = f"{mu:.2f}"
    raw_score_path = f'results/{graph_type}_n2v_raw_score_mu{mu_str}.pkl'
    with open(raw_score_path, 'wb') as file:
        pickle.dump(raw_score_mu, file)
    print(f"RAW_SCORE for mu={mu_str} saved to {raw_score_path}")

    raw_qf_path = f'results/{graph_type}_n2v_raw_qf_mu{mu_str}.pkl'
    with open(raw_qf_path, 'wb') as file:
        pickle.dump(raw_qf_mu, file)
    print(f"RAW_QF for mu={mu_str} saved to {raw_qf_path}")


def main():
    parser = argparse.ArgumentParser(description="Community detection on sparse and original networks.")
    parser.add_argument('--graph_type', type=str, choices=['ppm', 'lfr'], default='ppm', help="Random graph type (ppm or lfr)")
    parser.add_argument('--start_step', type=float, default=0.05, help="start_step")
    
    args = parser.parse_args()
    graph_type = args.graph_type
    start_step = args.start_step

    if graph_type == "ppm":
        end_step = 0.9
    elif graph_type == "lfr":
        end_step = 0.5
    step_size = 0.05
    MU = np.around(np.arange(start_step, end_step + 0.01, step_size), decimals=2)
    
    print(MU)
    
    print("程序已经在运行啦！")

    # 串行处理每个 mu
    for mu in MU:
        comprehensive_process(mu, graph_type)

    print("Computation completed")

if __name__ == "__main__":
    main()