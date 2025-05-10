from utils import *
from main import *

import networkx as nx
import random

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class NodeState:
    seq   : List[int]                 # individualisation sequence
    pi    : List[int]                  # colouring vector
    trace : Tuple                     # full event list from refinement
    tc    : List[int]                # target cell chosen


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

def relabel_random(G):
    perm = list(G.nodes())
    random.shuffle(perm)
    mapping = dict(zip(G.nodes(), perm))
    return nx.relabel_nodes(G, mapping), mapping

def equal_partitions(cells1, cells2, mapping):
    if len(cells1)!=len(cells2):
        return False
    trans = [sorted(mapping[v] for v in cell) for cell in cells1]
    return all(sorted(a)==sorted(b) for a,b in zip(trans, cells2))


def bfs_compare(G, H, mapping, max_depth=4):
    n     = G.number_of_nodes()
    pi0   = [0]*n
    alpha0= find_cells(pi0)

    # refinement and node state for the root
    def make_state(graph, pi_init, seq, trace=None):
        if not trace:
            trace = ()
        tc = ()  # filled later
        return NodeState(seq=seq, pi=pi_init, trace=trace, tc=tc)

    root_G = make_state(G,  refinement(G, pi0.copy(), alpha0.copy())[0], [])
    root_H = make_state(H,  refinement(H, pi0.copy(), alpha0.copy())[0], [])

    queue = [(root_G, root_H)]
    depth = 0

    while queue and depth <= max_depth:
        next_level=[]
        for nG, nH in queue:
            cellsG = find_cells(nG.pi)
            cellsH = find_cells(nH.pi)
            """print(nG.seq)
            print(nG.pi)
            print(cellsG)
            print(nG.trace)
            print(nH.trace)"""

            assert equal_partitions(cellsG,cellsH,mapping), f"partition diverged at seq {nG.seq}"
            #assert nG.trace == nH.trace, f"trace diverged at seq {nG.seq}"
            nG.tc  = target_cell_select(TreeNode(nG.seq), cellsG)
            nH.tc  = target_cell_select(TreeNode(nH.seq), cellsH)

            if not nG.tc:   # leaf reached
                continue

            assert sorted(nH.tc) == sorted([mapping[v] for v in nG.tc]), \
                   f"target cell rule broke at seq {nG.seq}"

            for i in range(len(nG.tc)):
                # ---- branch in original graph
                v = nG.tc[i]
                w = mapping[v]
                new_pi,new_trace,new_code = refinement(G,
                                    individualization(nG.pi, v),
                                    [[v]])
                childG = make_state(G, new_pi, nG.seq+[v],new_trace)

                new_piH,new_traceH,new_codeH = refinement(H,
                                    individualization(nH.pi, w),
                                    [[w]])
                childH = make_state(H, new_piH, nH.seq+[w],new_traceH)
                next_level.append((childG, childH))
        queue = next_level
        depth += 1



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
        if is_isomorphic(G, G_iso):
            correct_iso += 1
        if nx.is_isomorphic(G, G_iso):
            true_iso += 1
        if not is_isomorphic(G, G_noniso):
            correct_noniso += 1
        if not nx.is_isomorphic(G, G_noniso):
            true_noniso += 1
    #print(C_G.nodes)
    #print(C_G_iso.nodes)
    #print(C_G.edges)
    #print(C_G_iso.edges)
    print("iso acc:", correct_iso / amount)
    print("noniso acc:", correct_noniso / amount)
    print("iso true:", true_iso / amount)
    print("noniso true:", true_noniso / amount)

    n = 16
    regular = nx.random_regular_graph(3, n, seed=5)  # or any hard regular graph
    mapping = {i: (i * 7) % n for i in range(n)}
    regular_iso = nx.relabel_nodes(regular, mapping)

    bfs_compare(regular, regular_iso, mapping, max_depth=6)
    print("label-invariance test passed")