import time

from Exact_Algorithm.utils import *
import networkx as nx
import math
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics.perm_groups import PermutationGroup


def R(prefix, G, pi, level_best_trace=(tuple(), tuple(), tuple()), cells=None):
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
    t_0 = time.perf_counter()
    n = G.number_of_nodes()
    if n == 0:
        return nx.empty_graph(), ""
    Leaves = []
    root = TreeNode([])
    NodeQueue = deque([root])

    Total_Node_Count = 0
    Total_Refine_Time = 0
    Prune_by_PA = 0
    Prune_by_PC = 0
    Max_Width = 0
    leafDB = {}
    group = PermutationGroup(Permutation([i for i in range(n)]))
    pruned_seq: set[tuple] = set()  # subtree roots removed

    pi0 = color_init(G)
    pi_init, trace_init, _ = R(None, G, pi0)

    root.rc, root.trace, root.code, root.depth = pi_init, trace_init, 0, 0
    max_code = -math.inf

    if max(pi_init) == n - 1:
        Leaves.append(root)
    else:
        while NodeQueue:
            cur = NodeQueue.popleft()
            Total_Node_Count += 1

            def mark_pruned(seq):
                for k in range(1, len(seq) + 1):
                    pruned_seq.add(seq[:k])

            """if tuple(cur.sequence) in pruned_seq:
                continue

            leaf_pi, leaf_trace, leaf_code, leafDB, fillDB = \
                run_experimental_path(G, cur.rc, leafDB)
            if not fillDB and leaf_code in leafDB.keys():
                # print(leafDB[leaf_code][0])
                # print(leaf_pi)
                perm_a = Permutation(leafDB[leaf_code][0])
                perm_b = Permutation(leaf_pi)
                if perm_a != perm_b:
                    new_aut = perm_b * perm_a ** -1
                    # print(gen)
                    if Permutation([i for i in range(n)]) not in group.strong_gens:
                        if new_aut not in group.strong_gens:
                            gens = group.strong_gens + [new_aut]
                        else:
                            gens = group.strong_gens
                    else:
                        gens = [new_aut]
                    base = group.base if group.base != [] else list(range(n))
                    # print(base)
                    # print(gens)
                    group._base, group._strong_gens = group.schreier_sims_random(base=base, gens=gens)
                    # print(group.strong_gens)

                    prev_len = len(NodeQueue)
                    list_seq_g = []
                    for aut in group.strong_gens:
                        list_seq_g.append([aut(x) for x in cur.sequence])
                    # keep ν itself and any prefix that is lexicographically smaller

                    def deletable(node):
                        seq_prime = node.sequence
                        if cur.sequence < seq_prime:
                            return seq_prime in list_seq_g
                        return False

                    NodeQueue = deque(filter(lambda n: not deletable(n), NodeQueue))
                    Prune_by_PC += prev_len - len(NodeQueue)
"""
            cells = find_cells(cur.rc)
            TC = target_cell_select(cur, cells)
            if not TC:  # discrete
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

                t0 = time.perf_counter()
                child.rc, child.trace, new_code = R(v, G, child.lc, cells=cells)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                Total_Refine_Time += elapsed

                if not new_code:
                    child.code = cur.code
                else:
                    child.code = mixcode(cur.code, new_code)

                cur.children.append(child)
            if len(children[0].sequence) == n:
                for node in children:
                    Total_Node_Count += 1
                    Leaves.append(node)
            else:
                best = children[0]
                for c in children[1:]:
                    cmp = trace_cmp(c.trace, best.trace)
                    if cmp is None:  # undecided → keep both
                        continue
                    if cmp == 1:  # strictly greater
                        best = c
                winners = [c for c in children
                           if trace_cmp(c.trace, best.trace) in (0, None)]
                Prune_by_PA += len(children) - len(winners)
                Max_Width = max(Max_Width, len(winners))
                for c in winners:
                    NodeQueue.append(c)

    max_inv = max(L.trace for L in Leaves)
    cand = [L for L in Leaves if L.trace == max_inv]
    best = min(cand, key=lambda L: tuple(L.rc))
    t_1 = time.perf_counter()
    Total_Time = t_1 - t_0
    logstr = f"BFS Tree Generated has {Total_Node_Count} nodes, Tree depth is {best.depth}, " \
             f"Max Tree Width is {Max_Width}, Total refining time is {Total_Refine_Time:.4f}s, " \
             f"Total time is {Total_Time:.4f}s, " \
             f"Number of Prunes PA,PC: {Prune_by_PA, Prune_by_PC}"
    print(logstr)
    return graph_relabeling(G, best.rc), logstr


def canonical_representation(G):
    """ Computes the canonical representation (a string for comparison) of a graph G(V, E). """
    nodes = sorted(G.nodes())
    edges = sorted(tuple(sorted(edge)) for edge in G.edges())
    return str(nodes) + "|" + str(edges)


def is_isomorphic(G1, G2):
    return canonical_representation(canonical_form(G1)[0]) == canonical_representation(canonical_form(G2)[0]), canonical_form(G1)[1]


'''Testing Isomorphism'''
if __name__ == "__main__":
    n = 16
    G1 = nx.random_regular_graph(6, n)
    G2 = nx.relabel_nodes(G1, {i: (i * 7) % n for i in range(n)})
    G3 = nx.empty_graph()

    C1 = canonical_form(G1)[0]
    C2 = canonical_form(G2)[0]
    C3 = canonical_form(G3)[0]
    print(nx.is_isomorphic(C1, C2))
    print(C1.edges() == C2.edges())
    print(is_isomorphic(G1, G2))

    print(C1.nodes)
    print(C2.nodes)
    print(C1.edges)
    print(C2.edges)
