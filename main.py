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
    n = G.number_of_nodes()
    Leaves = []
    root   = TreeNode([])            # from utils
    NodeQueue = deque([root])

    # ---- initial colouring + trace
    pi0 = color_init(G)
    pi_init, trace_init = R(None, G, pi0)              # R now returns (pi, trace)
    root.rc, root.trace = pi_init, trace_init

    if max(pi_init) == n-1:
        Leaves.append(root)
    else:
        while NodeQueue:
            cur = NodeQueue.popleft()
            cells = find_cells(G, cur.rc)
            TC    = target_cell_select(cur, cells)
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

    # --- choose the lexicographically largest trace; break ties on branch
    max_inv = max(L.trace for L in Leaves)
    print(max_inv)
    cand = [L for L in Leaves if L.trace == max_inv]
    best = min(cand, key=lambda L: canonical_representation(
        graph_relabeling(G, L.rc)))
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
    G1 = nx.random_regular_graph(6, 16)
    G2 = nx.relabel_nodes(G1, {i: (i * 7) % 16 for i in range(16)})

    print(nx.is_isomorphic(canonical_form(G1), canonical_form(G2)))  # → True
    print(canonical_form(G1).edges() == canonical_form(G2).edges())  # → True
    print(is_isomorphic(G1,G2))



