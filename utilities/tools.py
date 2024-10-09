import os
import sys
import numpy as np
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
import clusim.sim as sim
from clusim.clustering import Clustering

# 获取当前目录的上一级目录
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.path.join(parent_dir),'EffectiveResistanceSampling'))
from EffectiveResistanceSampling.Network import *

def to_networkx(graph):
    """Convert a graph to NetworkX format."""
    G = nx.Graph()

    # 添加节点到图中
    G.add_nodes_from(range(graph.nodenum()))  # 如果需要添加其他属性可以在此修改

    # 添加边到图中
    edges = [(graph.E_list[i][0], graph.E_list[i][1], {'weight': graph.weights[i]}) for i in range(graph.edgenum())]
    G.add_edges_from(edges)

    return G

def euclid_membership(K, points):
    euc_kmeans = KMeans(n_clusters=K, n_init=10)
    euc_kmeans.fit(points)
    return euc_kmeans.labels_


def cosine_membership(K, points):
    normalized_points = normalize(points)
    cos_kmeans = KMeans(n_clusters=K, n_init=10)
    cos_kmeans.fit(normalized_points)
    return cos_kmeans.labels_


def calculate_score(evala, intr_list, K):
    """Calculate scores for clustering using NMI and ECSim."""
    return_val = []
    intr_clus = Clustering({i: [intr_list[i]] for i in range(len(intr_list))})
    evala_euclid_membership = euclid_membership(K, evala)
    evala_cosine_membership = cosine_membership(K, evala)

    return_val.append(normalized_mutual_info_score(evala_euclid_membership, intr_list, average_method='arithmetic'))
    return_val.append(normalized_mutual_info_score(evala_cosine_membership, intr_list, average_method='arithmetic'))

    evala_euclid_clustering = Clustering({i: [evala_euclid_membership[i]] for i in range(len(evala_euclid_membership))})
    evala_cosine_clustering = Clustering({i: [evala_cosine_membership[i]] for i in range(len(evala_cosine_membership))})

    return_val.append(sim.element_sim(intr_clus, evala_euclid_clustering, alpha=0.9))
    return_val.append(sim.element_sim(intr_clus, evala_cosine_clustering, alpha=0.9))

    return return_val


def load_graph(mu, rg, input_dir='graph'):
    mu_str = f"{mu:.2f}"
    file_path = os.path.join(input_dir, f'{rg}_graph_mu{mu_str}.pickle')
    with open(file_path, 'rb') as file:
        combined_data = pickle.load(file)
    return combined_data['graphs'], combined_data['memberships']