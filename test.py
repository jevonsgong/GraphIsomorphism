from utils import *
from main import *

import networkx as nx
import random


def generate_random_graph(n, p):
    """Generates an Erdos-Renyi random graph with n nodes and edge probability p."""
    return nx.erdos_renyi_graph(n, p)


def generate_isomorphic_variant(G):
    """Generates an isomorphic copy of G by applying a random permutation to the node labels."""
    nodes = list(G.nodes())
    permuted = random.sample(nodes, len(nodes))
    mapping = dict(zip(nodes, permuted))
    return nx.relabel_nodes(G, mapping)


def generate_non_isomorphic_variant(G):
    """
    Generates a non-isomorphic variant of G deterministically.
    The method uses a fixed modification:
      - If G is sparse (density < 0.5), try to add an edge (choose the first non-edge).
      - Otherwise, if G is dense, remove the first edge.
    If the chosen modification yields a graph isomorphic to G,
    try the opposite modification.
    """
    G_noniso = G.copy()
    density = nx.density(G)

    # Deterministic modification based on density
    if density < 0.5:
        # For a sparse graph, add the first non-edge (if available)
        non_edges = list(nx.non_edges(G_noniso))
        if non_edges:
            edge_to_add = non_edges[0]
            G_noniso.add_edge(*edge_to_add)
        else:
            # Fallback: remove an edge if no non-edges exist (unlikely for sparse graphs)
            edge_to_remove = list(G_noniso.edges())[0]
            G_noniso.remove_edge(*edge_to_remove)
    else:
        # For a dense graph, remove the first edge
        if G_noniso.number_of_edges() > 0:
            edge_to_remove = list(G_noniso.edges())[0]
            G_noniso.remove_edge(*edge_to_remove)
        else:
            # Fallback: add an edge if there are no edges (unlikely for dense graphs)
            non_edges = list(nx.non_edges(G_noniso))
            if non_edges:
                edge_to_add = non_edges[0]
                G_noniso.add_edge(*edge_to_add)

    # Verify non-isomorphism; if G_noniso is still isomorphic to G, try the opposite modification.
    if nx.is_isomorphic(G, G_noniso):
        # Revert and try the opposite: if we added an edge, try removing one; if we removed one, try adding one.
        G_noniso = G.copy()
        if density < 0.5:
            # Instead of adding an edge, remove the first edge.
            if G_noniso.number_of_edges() > 0:
                edge_to_remove = list(G_noniso.edges())[0]
                G_noniso.remove_edge(*edge_to_remove)
        else:
            # Instead of removing an edge, add the first non-edge.
            non_edges = list(nx.non_edges(G_noniso))
            if non_edges:
                edge_to_add = non_edges[0]
                G_noniso.add_edge(*edge_to_add)

    return G_noniso


def generate_infinite_test_cases():
    """
    A generator that yields an infinite sequence of test cases.
    Each test case is a tuple: (G, G_iso, G_noniso)

      - G: A randomly generated graph.
      - G_iso: An isomorphic copy of G.
      - G_noniso: A deterministically modified (and checked) non-isomorphic variant of G.
    """
    while True:
        # Randomly choose number of nodes and edge probability
        n = random.randint(4, 20)
        p = random.uniform(0.2, 0.8)

        # Generate a random graph G
        G = generate_random_graph(n, p)
        # Ensure G has at least one edge to allow modification
        if G.number_of_edges() == 0:
            continue

        # Create an isomorphic copy of G by applying a random permutation
        G_iso = generate_isomorphic_variant(G)

        # Create a non-isomorphic variant of G deterministically
        G_noniso = generate_non_isomorphic_variant(G)

        yield (G, G_iso, G_noniso)


def test_generator():
    """ Test if generator works correctly """
    test_case_generator = generate_infinite_test_cases()

    # Example: Print 5 test cases
    for i in range(5):
        G, G_iso, G_noniso = next(test_case_generator)
        print(f"Test case {i + 1}:")
        print(f"  G nodes: {G.nodes()}  edges: {list(G.edges())}")
        print(f"  G_iso is isomorphic to G? {nx.is_isomorphic(G, G_iso)}")
        print(f"  G_noniso is isomorphic to G? {nx.is_isomorphic(G, G_noniso)}")
        print("-" * 40)


if __name__ == "__main__":
    test_case_generator = generate_infinite_test_cases()
    correct_iso = 0
    true_iso = 0
    correct_noniso = 0
    true_noniso = 0
    amount = 100
    for i in range(amount):
        G, G_iso, G_noniso = next(test_case_generator)
        C_G = canonical_form(G)
        C_G_iso = canonical_form(G_iso)
        C_G_noniso = canonical_form(G_noniso)
        if C_G == C_G_iso:
            correct_iso += 1
        if nx.is_isomorphic(G, G_iso):
            true_iso += 1
        if C_G != C_G_noniso:
            correct_noniso += 1
        if not nx.is_isomorphic(G, G_noniso):
            true_noniso += 1
    print("iso acc:", correct_iso / amount)
    print("noniso acc:", correct_noniso / amount)
    print("iso true:", true_iso / amount)
    print("noniso true:", true_noniso / amount)

