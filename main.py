from collections import deque, defaultdict
from utils import refinement, find_cells, individualization, TreeNode, color_init, target_cell_select, N
import networkx as nx


def R(prefix, G, pi, cells=None):
    """ General function for computing the refined coloring. """
    if not prefix:
        if pi:
            return refinement(G, pi, find_cells(G, pi))
        else:
            return refinement(G, pi, cells)
    return refinement(G, individualization(pi, prefix), [[prefix]])


def canonical_labeling(G):
    """ Computes the canonical labeling of a graph G(V, E). """
    V = list(G.nodes())
    Leaves = []

    root = TreeNode([])  # Root is an empty sequence
    NodeQueue = deque([root])

    pi_0 = color_init(G)
    pi_init = R(None, G, pi_0)
    root.lc = pi_0
    root.rc = pi_init

    while NodeQueue:
        cur_node = NodeQueue.popleft()
        cur_cells = find_cells(G, cur_node.rc)
        TC = target_cell_select(cur_cells)
        # print("NodeQueue:", NodeQueue)
        # print("Color:", cur_node.rc)
        # print("TC", TC)
        # print("Leaves:", Leaves)
        for i, v in enumerate(TC):
            if v not in cur_node.sequence:
                NextNode = TreeNode(cur_node.sequence + [v])
                cur_node.children.append(NextNode)
                NextNode.lc = cur_node.rc
                NextNode.rc = R(v, G, NextNode.lc, cells=cur_cells)
                NextNode.traces = N(G, NextNode.rc)
                if max(NextNode.rc) == len(V) - 1:  # Check if refined color is discrete
                    Leaves.append(NextNode)
                else:
                    NodeQueue.append(NextNode)

    BestNode = max(Leaves, key=lambda node: node.traces)
    C_G = BestNode.rc

    return C_G


'''Testing Isomorphism'''
if __name__ == "__main__":
    def generate_test_graphs():
        # Graph 1 (Base graph)
        G1 = nx.Graph()
        G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])  # A simple 4-node cycle with a chord

        # Graph 2 (Isomorphic to G1, relabeled)
        mapping = {0: 3, 1: 2, 2: 1, 3: 0}  # Relabel nodes
        G2 = nx.relabel_nodes(G1, mapping)

        # Graph 3 (Non-isomorphic: different structure)
        G3 = nx.Graph()
        G3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])  # A path of 5 nodes (not a cycle)

        G4 = nx.Graph()
        G4.add_edges_from([(0,1), (1,2), (2,3), (3,0)])

        return G1, G2, G3, G4


    # Generate and print test graphs
    G1, G2, G3, G4 = generate_test_graphs()

    C1 = canonical_labeling(G1)
    C2 = canonical_labeling(G2)
    C3 = canonical_labeling(G3)
    C4 = canonical_labeling(G4)

    print("C1:", C1)
    print("C2:", C2)
    print("C3:", C3)
    print("C4:", C4)
