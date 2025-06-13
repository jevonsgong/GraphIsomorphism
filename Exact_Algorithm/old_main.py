import time

from Exact_Algorithm.old_util import *
import networkx as nx
import math
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup


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
        return nx.empty_graph(), ""
    Leaves = []
    root = TreeNode([])
    NodeQueue = deque([root])

    Total_Node_Count = 0
    Total_Refine_Time = 0
    Prune_by_PA = 0
    Prune_by_PB = 0
    Prune_by_PC = 0
    Max_Width = 0
    leafDB = {}
    group = PermutationGroup(Permutation([i for i in range(n)]))

    pi0 = color_init(G)
    pi_init, trace_init, _ = R(None, G, pi0)

    root.rc, root.trace, root.code, root.depth = pi_init, trace_init, 0, 0
    level_best_code: dict[int, int] = {}  # depth -> best code
    level_best_trace: dict[int, tuple] = {}  # depth -> best trace
    level_best_node: dict[int, TreeNode] = {}  # depth -> keeper
    pruned_seq: set[tuple] = set()  # subtree roots removed
    max_code = -math.inf

    if max(pi_init) == n - 1:
        Leaves.append(root)
    else:
        while NodeQueue:
            cur = NodeQueue.popleft()

            cells = find_cells(cur.rc)
            TC = target_cell_select(cur, cells)
            if not TC:            # discrete
                Leaves.append(cur)
                continue
            children = []
            for v in TC:
                if v in cur.sequence:
                    continue
                seq = cur.sequence + [v]
                child = TreeNode(seq)
                children.append(child)
                child.parent = cur
                child.lc, child.depth = cur.rc, cur.depth + 1
                child.rc, child.trace, new_code = R(v, G, child.lc, cells=cells)
                if not new_code:
                    child.code = cur.code
                else:
                    child.code = mixcode(cur.code, new_code)

                cur.children.append(child)
            if len(children[0].sequence) == n:
                for node in children:
                    Leaves.append(node)
            else:
                best = children[0]
                for c in children[1:]:
                    cmp = trace_cmp(c.trace, best.trace)  # 64-bit code optional
                    if cmp is None:  # undecided → keep both
                        continue
                    if cmp == 1:  # strictly greater
                        best = c
                winners = [c for c in children
                           if trace_cmp(c.trace, best.trace) in (0, None)]
                for c in winners:
                    NodeQueue.append(c)

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
    return canonical_representation(canonical_form(G1))==canonical_representation(canonical_form(G2))

'''Testing Isomorphism'''
if __name__ == "__main__":
    n = 16
    G1 = nx.random_regular_graph(6, n)
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