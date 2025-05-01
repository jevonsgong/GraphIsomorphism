from utils import *
import networkx as nx


def R(prefix, G, pi, cells=None):
    """ General function for computing the refined coloring. """
    if prefix is None:
        if pi:
            return refinement(G, pi, find_cells(pi))
        else:
            return refinement(G, pi, cells)
    return refinement(G, individualization(pi, prefix), [[prefix]])


def graph_relabeling(G, pi):
    nodes_sorted = sorted(G.nodes(), key=lambda v: (pi[v], v))
    mapping = {old: new for new, old in enumerate(nodes_sorted)}
    return nx.relabel_nodes(G, mapping)


def canonical_form(G):
    n = G.number_of_nodes()
    Leaves = []
    root = TreeNode([])
    NodeQueue = deque([root])

    pi0 = color_init(G)
    pi_init, trace_init = R(None, G, pi0)
    root.rc, root.trace = pi_init, trace_init

    if max(pi_init) == n-1:
        Leaves.append(root)
    else:
        while NodeQueue:
            cur = NodeQueue.popleft()
            cells = find_cells(cur.rc)
            TC = target_cell_select(cur, cells)
            if not TC:            # discrete
                Leaves.append(cur)
                continue
            for v in TC:
                if v in cur.sequence:
                    continue
                seq = cur.sequence + [v]
                child = TreeNode(seq)
                child.lc = cur.rc
                child.rc, child.trace = R(v, G, child.lc, cells=cells)
                cur.children.append(child)
                if max(child.rc) == n-1:
                    Leaves.append(child)
                else:
                    NodeQueue.append(child)

    best = max(Leaves, key=lambda node: (node.trace, tuple(node.sequence)))
    return graph_relabeling(G, best.rc)

def canonical_representation(G):
    """ Computes the canonical representation (a string for comparison) of a graph G(V, E). """
    nodes = sorted(G.nodes())
    edges = sorted(tuple(sorted(edge)) for edge in G.edges())
    return str(nodes) + "|" + str(edges)

def is_isomorphic(G1, G2):
    return canonical_representation(canonical_form(G1))==canonical_representation(canonical_form(G2))

'''Testing Isomorphism'''
if __name__ == "__main__":
    n = 16
    G1 = nx.random_regular_graph(3, n, seed=5)
    G2 = nx.relabel_nodes(G1, {i: (i * 7) % n for i in range(n)})

    C1 = canonical_form(G1)
    C2 = canonical_form(G2)
    print(nx.is_isomorphic(C1, C2))
    print(C1.edges() == C2.edges())
    print(is_isomorphic(G1,G2))

    print(C1.nodes)
    print(C2.nodes)
    print(C1.edges)
    print(C2.edges)


