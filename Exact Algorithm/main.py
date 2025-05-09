from utils import *
import networkx as nx


def R(prefix, G, pi, cells=None):
    """ General function for computing the refined coloring. """
    if prefix is None:
        if pi:
            return refinement(G, pi, find_cells(G, pi))
        else:
            return refinement(G, pi, cells)
    return refinement(G, individualization(pi, prefix), [[prefix]])


def graph_relabeling(G, pi):
    nodes_sorted = sorted(G.nodes(), key=lambda v: (pi[v], v))
    mapping = {old: new for new, old in enumerate(nodes_sorted)}
    return nx.relabel_nodes(G, mapping)


def canonical_form(G):
    """ Computes the canonical form of a graph G(V, E). """
    n = G.number_of_nodes()
    Leaves = []
    root = TreeNode([])
    NodeQueue = deque([root])

    # Compute the initial coloring and first refinement.
    pi_0 = color_init(G)
    pi_init = R(None, G, pi_0)
    root.lc = pi_0
    root.rc = pi_init
    root.invariant = node_invariant(G, root.rc, root.sequence)

    # End if the initial refined partition is discrete
    if max(root.rc) == n:
        Leaves.append(root)
    else:
        # Breadth-first search on the tree.
        while NodeQueue:
            cur_node = NodeQueue.popleft()
            cells = find_cells(G, cur_node.rc)
            TC = target_cell_select(cur_node, cells)
            # If no non-singleton cell was found, then the partition is discrete.
            if not TC:
                Leaves.append(cur_node)
                continue
            # For each vertex in the chosen target cell, create a child node by individualizing it.
            for v in TC:
                if v not in cur_node.sequence:
                    new_seq = cur_node.sequence + [v]
                    child_node = TreeNode(new_seq)
                    child_node.parent = cur_node
                    cur_node.children.append(child_node)
                    child_node.lc = cur_node.rc
                    # Use R to complete the IR refinement
                    child_node.rc = R(v, G, child_node.lc, cells=cells)
                    child_node.invariant = node_invariant(G, child_node.rc, new_seq)
                    # If the refined partition is discrete, add to Leaves; otherwise, add for further expansion.
                    if max(child_node.rc) == n:
                        Leaves.append(child_node)
                    else:
                        NodeQueue.append(child_node)

    # Choose the leaf with maximum invariant and, if tied, choose the lexicographically smallest branch.
    BestNode = max(Leaves, key=lambda node: (node.invariant, tuple(node.sequence)))
    C_label = BestNode.rc
    C_G = graph_relabeling(G, C_label)
    return C_G

def canonical_representation(G):
    """ Computes the canonical representation (a string for comparison) of a graph G(V, E). """
    nodes = sorted(G.nodes())
    edges = sorted(tuple(sorted(edge)) for edge in G.edges())
    return str(nodes) + "|" + str(edges)

def is_isomorphic(G1, G2):
    return canonical_representation(canonical_form(G1))==canonical_representation(canonical_form(G2))

'''Testing Isomorphism'''
if __name__ == "__main__":
    def generate_test_graphs():
        # Graph 1 (Base graph)
        G1 = nx.Graph()
        G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

        # Graph 2 (Isomorphic to G1, relabeled)
        mapping = {0: 3, 1: 2, 2: 1, 3: 0}  # Relabel nodes
        G2 = nx.relabel_nodes(G1, mapping)

        # Graph 3 (Non-isomorphic: different structure)
        G3 = nx.Graph()
        G3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

        G4 = nx.Graph()
        G4.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        return G1, G2, G3, G4


    # Generate and print test graphs
    G1, G2, G3, G4 = generate_test_graphs()

    C1 = canonical_form(G1)
    C2 = canonical_form(G2)
    # C3 = canonical_form(G3)
    #C4 = canonical_form(G4)

    print(is_isomorphic(G1,G2))
    #print(graphs_equal(C1,C4))
    #print(C1 == C4)
    print(C1.nodes)
    print(C2.nodes)
    print(C1.edges)
    print(C2.edges)


    root = TreeNode([])
    node1 = TreeNode([0])
    node2 = TreeNode([3])
    node1.parent = root
    node2.parent = root



