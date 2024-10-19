# Set the range for mixing parameters
import numpy as np

num_processor = 20
sample_count = 50

step_total = 10
step_size = 0.05
MU = np.around(np.arange(step_size, step_size * step_total + 0.01, step_size), decimals=2)

# Specify Parameters
n = 1000
tau1 = 2  # Power-law exponent for the degree distribution
tau2 = 1.1  # Power-law exponent for the community size distribution
avg_deg = 25  # Average Degree
max_deg = int(0.1 * n)  # Max Degree
min_commu = 60  # Min Community Size
max_commu = int(0.1 * n)  # Max Community Size


import os
import pickle
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph


def gene_lfr_graph_mu(mu, sample_count=sample_count, output_dir='graph'):
    """
    Generates LFR benchmark graphs for a given mixing parameter (mu) and saves the graphs and memberships to a file.

    Parameters:
        mu (float): Mixing parameter.
        sample_count (int): Number of graph samples to generate.
        output_dir (str): Directory where the generated graphs and memberships are saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    graphs = []
    memberships = []

    for i in range(sample_count):
        G = LFR_benchmark_graph(
            n, tau1, tau2, mu, average_degree=avg_deg, max_degree=max_deg,
            min_community=min_commu, max_community=max_commu, seed=2024
        )

        # Convert to undirected graph and remove self-loops
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Get the intrinsic communities and memberships
        intrinsic_communities = {frozenset(G.nodes[v]["community"]) for v in G}
        intrinsic_membership = {}
        for node in range(G.number_of_nodes()):
            for index, inner_set in enumerate(intrinsic_communities):
                if node in inner_set:
                    intrinsic_membership[node] = index
                    break
        intrinsic_membership = list(intrinsic_membership.values())
        intrinsic_membership = np.array(intrinsic_membership)

        # Append the graph and membership to the lists
        graphs.append(G)
        memberships.append(intrinsic_membership)

    # Save all graphs and memberships into a single file
    combined_data = {
        'graphs': graphs,
        'memberships': memberships
    }

    mu_str = f"{mu:.2f}"
    file_path = os.path.join(output_dir, f'lfr_graph_original_mu{mu_str}.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(combined_data, file)

    print(f'Saved all graphs and memberships for mu={mu} to {file_path}')


# Generate LFR graphs in parallel using ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    num_workers = num_processor  # 指定 worker 的数量
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(gene_lfr_graph_mu, MU))
