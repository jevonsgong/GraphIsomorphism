from utils import *
import networkx as nx
import math


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
    if n == 0:
        return nx.empty_graph()
    Leaves = []
    root = TreeNode([])
    NodeQueue = deque([root])

    pi0 = color_init(G)
    pi_init, trace_init, _ = R(None, G, pi0)
    root.rc, root.trace, root.code, root.depth = pi_init, trace_init, 0, 0
    level_best_code = {}  # depth -> code
    level_best_trace = {}  # depth -> trace
    best_keeper = {}  # depth -> node
    pruned_nodes = []
    max_code = -math.inf

    if max(pi_init) == n - 1:
        Leaves.append(root)
    else:
        while NodeQueue:
            cur = NodeQueue.popleft()

            """#  if current node is in a subtree that should be discarded, skip it
            #  (A better subtree a found later than this node is appended to Queue)
            if cur.parent in pruned_nodes:
                continue
            #  Pruning PA and PB
            code = cur.code
            tr = cur.trace
            depth = cur.depth
            if depth in level_best_code:
                best_code = level_best_code[depth]
                if code < best_code:  # PA
                    continue  # discard subtree
                if code > best_code:
                    pruned_nodes.append(best_keeper[depth])
                    best_keeper[depth] = cur
                    level_best_code[depth] = code
                else:
                    best_trace = level_best_trace[depth]
                    if tr < best_trace:  # PA
                        continue  # discard subtree
                    if tr > best_trace:
                        pruned_nodes.append(best_keeper[depth])
                        best_keeper[depth] = cur
                        level_best_trace[depth] = tr
                    else:
                        if cur.sequence > best_keeper[depth].sequence:  # PB
                            continue
                        elif cur.sequence < best_keeper[depth].sequence:
                            pruned_nodes.append(best_keeper[depth])
                            best_keeper[depth] = cur
            else:
                level_best_code[depth] = code
                level_best_trace[depth] = tr
                best_keeper[depth] = cur"""

            cells = find_cells(cur.rc)
            TC = target_cell_select(cur, cells)
            if not TC:  # discrete
                Leaves.append(cur)
                continue
            for v in TC:
                if v in cur.sequence:
                    continue
                seq = cur.sequence + [v]
                child = TreeNode(seq)
                child.parent = cur
                child.lc, child.depth = cur.rc, cur.depth + 1
                child.rc, child.trace, new_code = R(v, G, child.lc, cells=cells)
                if not new_code:
                    child.code = cur.code
                else:
                    child.code = mixcode(cur.code, new_code)
                cur.children.append(child)
                if max(child.rc) == n - 1:
                    if child.code > max_code:
                        Leaves = [child]
                        max_code = child.code
                    elif child.code == max_code:
                        Leaves.append(child)
                else:
                    NodeQueue.append(child)

    max_inv = max(L.trace for L in Leaves)
    cand = [L for L in Leaves if L.trace == max_inv]
    best = min(cand, key=lambda L: tuple(L.rc))
    return graph_relabeling(G, best.rc)


def canonical_representation(G):
    """ Computes the canonical representation (a string for comparison) of a graph G(V, E). """
    nodes = sorted(G.nodes())
    edges = sorted(tuple(sorted(edge)) for edge in G.edges())
    return str(nodes) + "|" + str(edges)


def is_isomorphic(G1, G2):
    return canonical_representation(canonical_form(G1)) == canonical_representation(canonical_form(G2))


'''Testing Isomorphism'''
if __name__ == "__main__":
    n = 16
    G1 = nx.random_regular_graph(6, n)
    G2 = nx.relabel_nodes(G1, {i: (i * 7) % n for i in range(n)})
    G3 = nx.empty_graph()

    C1 = canonical_form(G1)
    C2 = canonical_form(G2)
    C3 = canonical_form(G3)
    print(nx.is_isomorphic(C1, C2))
    print(C1.edges() == C2.edges())
    print(is_isomorphic(G1, G2))

    print(C1.nodes)
    print(C2.nodes)
    print(C1.edges)
    print(C2.edges)
