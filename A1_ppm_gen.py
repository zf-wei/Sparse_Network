import numpy as np
import networkx as nx
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

num_processor = 20
sample_count = 50             # Number of graph samples to generate

# Set the range for mixing parameters
step_total = 18
step_size = 0.05
MU = np.around(np.arange(step_size, step_size * step_total + 0.01, step_size), decimals=2)

# Specify Parameters
number_of_comm = 15           # Number of communities
comm_size = 68                # Size of communities
n = number_of_comm * comm_size # Total number of nodes
deg_avg = 25                  # Average degree of the whole network


def generate_ppm(q, comm_size, deg_avg, mu):
    """
    Generates a graph based on the Planted Partition Model (PPM).

    Parameters:
        q (int): Number of communities.
        comm_size (int): Size of each community.
        deg_avg (int): Average degree of the network.
        mu (float): Mixing parameter (controls between-community edges).

    Returns:
        G (networkx.Graph): The generated PPM graph.
    """
    n = q * comm_size  # Total number of nodes
    # Calculate probabilities for edges between and within communities
    p_out = mu * deg_avg / n
    p_in = (deg_avg - (n - n / q) * p_out) / (n / q - 1)

    if p_out < 0 or p_out > 1 or p_in < 0 or p_in > 1:
        raise ValueError("Calculated probability is out of the valid range (0, 1).")

    sizes = [n // q] * q  # List of community sizes (equal-sized communities)

    # Probability matrix (within and between blocks)
    p_matrix = [[p_in if i == j else p_out for j in range(q)] for i in range(q)]

    # Generate the PPM graph using the stochastic block model (SBM) generator
    G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
    return G


def gene_ppm_graph_mu(mu, sample_count=sample_count, output_dir='graph_original'):
    """
    Generates PPM graphs for a given mixing parameter (mu) and saves the graphs and memberships to a file.

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
        G = generate_ppm(number_of_comm, comm_size, deg_avg, mu)

        # Remove self-loops
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Get intrinsic membership (community assignment)
        intrinsic_membership = list(nx.get_node_attributes(G, 'block').values())
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
    file_path = os.path.join(output_dir, f'ppm_graph_original_mu{mu_str}.pickle')
    with open(file_path, 'wb') as file:
        pickle.dump(combined_data, file)

    print(f'Saved all generated graphs and memberships for mu={mu_str} to {file_path}')


# Generate PPM graphs in parallel using ProcessPoolExecutor
if __name__ == "__main__":
    num_workers = num_processor  # 指定 worker 的数量
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(gene_ppm_graph_mu, MU))
