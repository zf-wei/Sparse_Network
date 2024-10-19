import os
import sys
import numpy as np
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# 获取当前目录的上一级目录
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.append(os.path.join(os.path.join(parent_dir),'EffectiveResistanceSampling'))
# from EffectiveResistanceSampling.Network import *

def to_networkx(graph):
    """Convert a graph to NetworkX format."""
    G = nx.Graph()

    # 添加节点到图中
    G.add_nodes_from(range(graph.nodenum()))  # 如果需要添加其他属性可以在此修改

    # 添加边到图中
    edges = [(graph.E_list[i][0], graph.E_list[i][1], {'weight': graph.weights[i]}) for i in range(graph.edgenum())]
    G.add_edges_from(edges)

    return G

def get_euclid_membership(K, points):
    euc_kmeans = KMeans(n_clusters=K, n_init=10)
    euc_kmeans.fit(points)
    return euc_kmeans.labels_


def get_cosine_membership(K, points):
    normalized_points = normalize(points)
    cos_kmeans = KMeans(n_clusters=K, n_init=10)
    cos_kmeans.fit(normalized_points)
    return cos_kmeans.labels_


def load_graph(mu, rg, delete_type='original'):
    mu_str = f"{mu:.2f}"
    input_dir = f"graph_{delete_type}"
    file_path = os.path.join(input_dir, f'{rg}_graph_{delete_type}_mu{mu_str}.pickle')

    with open(file_path, 'rb') as file:
        combined_data = pickle.load(file)

    graphs = combined_data['graphs']
    memberships = combined_data.get('memberships')  # Use .get() to avoid KeyError

    if memberships is not None:
        return graphs, memberships  # Return both if memberships exist
    else:
        return graphs  # Return only graphs if memberships do not exist

def load_graph_only(mu, rg, delete_type='original'):
    mu_str = f"{mu:.2f}"
    input_dir = f"graph_{delete_type}"
    file_path = os.path.join(input_dir, f'{rg}_graph_{delete_type}_mu{mu_str}.pickle')

    with open(file_path, 'rb') as file:
        combined_data = pickle.load(file)

    graphs = combined_data['graphs']
    return graphs  # Return only graphs if memberships do not exist