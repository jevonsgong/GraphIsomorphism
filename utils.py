from collections import deque, defaultdict, OrderedDict
import random
import hashlib
import networkx as nx

'''BFS Tree'''


class TreeNode:
    def __init__(self, sequence):
        self.sequence = sequence  # Sequence of vertices
        self.lc = None  # Last color
        self.rc = None  # Refined color
        self.traces = None  # Traces/Node Invariant
        self.children = []
        self.parent = []
        self.N = None #  Node Invariant Value


'''Individualization-Refinement Implementation'''


def color_init(G):
    """ Initialize coloring """
    pi = [0] * nx.number_of_nodes(G)
    return pi


def classify_by_edges(G, X, W):
    """
    Classify vertices in X into groups based on the number of edges to W.
    Returns the groups in a canonical order: sorted by the count,
    and within each group, vertices are sorted.
    """
    edge_count = defaultdict(list)

    for v in X:
        count = sum(1 for w in W if w in G.neighbors(v))
        edge_count[count].append(v)

    # Canonically sort the groups: sort keys and sort vertices in each group
    groups = []
    for count in sorted(edge_count.keys()):
        group = sorted(edge_count[count])
        groups.append(group)

    return groups


def replace_cell(cells, X, new_cells):
    """ Replace a cell X in partition with new cells. """
    for i,cell in enumerate(cells):
        if cell == X:
            del(cells[i])
            for j,new_cell in enumerate(new_cells):
                cells.insert(i+j,new_cell)
    return 0


def append_largest_except_one(alpha, new_cells):  # checked
    """ Append all but the largest cell in new_cells to alpha. """
    if not new_cells:
        return alpha

    largest_cell = max(new_cells, key=len)
    for cell in new_cells:
        if cell != largest_cell:
            alpha.append(cell)

    return alpha


def individualization(pi, w):
    """ Perform the Individualization step I(pi, w) -> pi' """
    pi_prime = pi.copy()
    for v in range(len(pi)):
        if pi[v] < pi[w] or v == w:
            continue
        else:
            pi_prime[v] = pi[v] + 1

    #pi_prime = pi.copy()
    #pi_prime[w] = max(pi) + 1  # Assign new color
    return pi_prime



def refinement(G, pi, alpha):
    """ Perform the Refinement step F(G, pi, alpha) """
    cells = find_cells(G, pi)
    while alpha and max(pi) != len(pi) - 1:
        W = alpha.pop(0)
        for X in cells:
            groups = classify_by_edges(G, X, W)
            replace_cell(cells, X, groups)

            if X in alpha:
                replace_cell(alpha, X, groups)
            else:
                append_largest_except_one(alpha, groups)

    return find_color(G, cells)


def find_cells(G, pi):
    """Transform the color vector pi into a canonical partition.
       Cells are sorted by their color, and within each cell, vertices are sorted.
       Then, sort the list of cells by (color, min(cell))."""
    cell_dict = {}
    for node, color in enumerate(pi):
        cell_dict.setdefault(color, []).append(node)
    # Sort vertices within each cell
    for color in cell_dict:
        cell_dict[color].sort()
    # Create list of (color, cell) pairs, then sort by (color, min(cell))
    cells = [(color, cell) for color, cell in cell_dict.items()]
    cells.sort(key=lambda x: (x[0], x[1][0] if x[1] else float('inf')))
    # Return just the list of cells (dropping the color keys)
    return [cell for _, cell in cells]


def find_color(G, cells):
    """ Transform from cells to color """
    pi = [0] * G.number_of_nodes()
    for k, cell in enumerate(cells):
        for v in cell:
            pi[v] = k
    return pi


def target_cell_select(cells):
    """Select the target cell in a canonical way.
       Tie-break by the smallest vertex in the cell."""
    return max(cells, key=len)


'''Node Invariant/Traces, Graph Sorting Implementation'''

FUZZ_CONSTANTS = [0o37541, 0o61532, 0o05257, 0o26416]


def fuzz1(x):
    """
    Mimics the C++ FUZZ1 macro.
    Given an integer x, returns x XOR-ed with a constant selected based on the lower two bits of x.
    """
    return x ^ FUZZ_CONSTANTS[x & 3]


def mash_comm(l, i):
    """
    Mimics the C++ MASHCOMM macro.
    'l' is the current invariant, 'i' is the new value to mix in.
    """
    return l + fuzz1(i)


def compute_invariant(last_invariant, cur_trace):
    """ Computes N """
    invariant = last_invariant + fuzz1(cur_trace)
    return invariant


def compute_traces(G, pi, last_trace):
    """ Compute Traces """
    cells = find_cells(G, pi)
    trace = last_trace
    for i,cell in enumerate(cells):
        cell_value = mash_comm(i,min(cell))
        trace = mash_comm(trace, cell_value)
    return trace


def hash_graph(G):
    """ Computes a deterministic hash for the graph G using sorted adjacency lists """
    edge_list = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    edge_str = "".join(f"{u}-{v}," for u, v in edge_list)
    return hashlib.sha256(edge_str.encode()).hexdigest()


def sort_partitions_by_quotient(G, partitions):
    """ Sorts partitions based on the hash of their quotient graphs """
    partition_hashes = [(hash_graph(nx.quotient_graph(G, p)), p) for p in partitions]
    partition_hashes.sort()  # Lexicographic sorting
    return [p for _, p in partition_hashes]


def graphs_equal(graph1, graph2):
    """Check if graphs are equal.

    Equality here means equal as Python objects (not isomorphism).
    Node, edge and graph data must match.

    Parameters
    ----------
    graph1, graph2 : graph

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    return (
        graph1.adj == graph2.adj
        and graph1.nodes == graph2.nodes
        and graph1.graph == graph2.graph
    )

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

    pi_0 = color_init(G)
    print("example initial labeling:", [i for i in range(G.number_of_nodes())])
    print("example initial color:", pi_0)
    print("example cells:", find_cells(G, pi_0))

    pi_i = refinement(G, pi_0, find_cells(G, pi_0))
    print("example initial refined color:", pi_i)

    w = target_cell_select(find_cells(G, pi_i))[0]
    pi_1I = individualization(pi_i, w)
    print("example first IR individualized color:", pi_1I)
    pi_1R = refinement(G, pi_1I, [[w]])
    print("example first IR refined color:", pi_1R)

    w = target_cell_select(find_cells(G, pi_1R))[0]
    pi_2I = individualization(pi_1R, w)
    print("example Second IR individualized color:", pi_2I)
    pi_2R = refinement(G, pi_2I, [[w]])
    print("example Second IR refined color:", pi_2R)
    final_cell = find_cells(G, pi_2R)
    print("example final cell:", final_cell)



