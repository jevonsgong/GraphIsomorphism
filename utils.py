from collections import deque, defaultdict, OrderedDict, Counter
import random
import hashlib
import networkx as nx

'''BFS Tree'''


class TreeNode:
    def __init__(self, sequence):
        self.sequence = sequence  # Sequence of vertices
        self.lc = None  # Last color
        self.rc = None  # Refined color
        self.trace = None  # Traces/Node Invariant
        self.target_cell = None  # Target cell
        self.children = []
        self.parent = None
        self.N = None  # Node Invariant Value


class TraceRecorder:
    def __init__(self):
        self.events = []  # list of (cell_id, sizes_tuple, edge_sig)

    def record(self, cell_id, sizes, edge_sig):
        """
        cell_id   : the colour id of the cell being split
        sizes     : tuple of fragment sizes (order preserved)
        edge_sig  : tuple of (#edges_from_fragment0_to_each_colour, …)
                    flattened
        """
        self.events.append((cell_id, sizes, edge_sig))

    # Convertible to an immutable tuple so we may use it as a key
    def freeze(self):
        return tuple(self.events)


def cell_edge_signature(G, X, cells):
    neigh = {v: set(G[v]) for v in X}
    index = {v: i for i, cell in enumerate(cells) for v in cell}

    deg_vectors = []
    for v in sorted(X):
        vec = [0] * len(cells)
        for u in neigh[v]:
            vec[index[u]] += 1
        deg_vectors.append(tuple(vec))
    return tuple(sorted(deg_vectors))


'''Individualization-Refinement Implementation'''


def color_init(G):
    """ Initialize coloring """
    pi = [0] * nx.number_of_nodes(G)
    return pi


def classify_by_edges(G, X, W):
    """
    Classify vertices in X into groups based on the number of edges to W.
    Returns the groups in a canonical order.(By edge count)
    """
    edge_count = defaultdict(list)

    for v in X:
        count = sum(1 for w in W if w in G.neighbors(v))
        edge_count[count].append(v)

    groups = []
    for count in sorted(edge_count.keys()):
        group = sorted(edge_count[count])
        groups.append(group)

    return groups


def replace_cell(cells, X, new_cells):
    """ Replace a cell X in partition with new cells. """
    for i, cell in enumerate(cells):
        if cell == X:
            del (cells[i])
            for j, new_cell in enumerate(new_cells):
                cells.insert(i + j, new_cell)


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

    # pi_prime = pi.copy()
    # pi_prime[w] = max(pi) + 1  # Assign new color
    return pi_prime


def refinement(G, pi, alpha):
    """
    Perform the classical equitable refinement BUT
    collect a TraceRecorder describing every split event.
    """
    recorder = TraceRecorder()
    cells = find_cells(pi)
    colour_map = {v: pi[v] for v in range(len(pi))}
    num_cols = max(pi) + 1 if pi else 0

    while alpha and max(pi) != len(pi) - 1:
        W = alpha.pop(0)
        for X in cells:
            groups = classify_by_edges(G, X, W)

            if len(groups) > 1:
                replace_cell(cells, X, groups)
                # --- record this split event --------------------------
                sizes = tuple(len(g) for g in groups)
                edge_sig = cell_edge_signature(G, X, cells)
                recorder.record(pi[X[0]], sizes, edge_sig)
                # ------------------------------------------------------

            if X in alpha:
                replace_cell(alpha, X, groups)
            else:
                append_largest_except_one(alpha, groups)

        # rebuild colour map quickly
        pi = find_color(G, cells)
        colour_map = {v: pi[v] for v in range(len(pi))}
        num_cols = max(pi) + 1

    return pi, recorder.freeze()


def find_cells(pi):
    """Transform the color vector pi into a canonical partition.
       Cells are sorted by their color, and within each cell, vertices are sorted."""
    cell_dict = defaultdict(list)
    for v, c in enumerate(pi):
        cell_dict[c].append(v)

    # build sortable signatures
    sigs = []
    for old_color, verts in cell_dict.items():
        verts.sort()
        multiset = tuple(pi[v] for v in verts)
        sig = (len(verts), multiset)
        sig += (tuple(verts),)
        sigs.append((sig, verts))

    sigs.sort(key=lambda x: x[0])

    # produce ordered cells and new colour vector
    new_pi = [0] * len(pi)
    cells = []
    for new_color, (_, verts) in enumerate(sigs):
        cells.append(verts)
        for v in verts:
            new_pi[v] = new_color
    return cells


def find_color(G, cells):
    """ Transform from cells to color """
    pi = [0] * G.number_of_nodes()
    for k, cell in enumerate(cells):
        for v in cell:
            pi[v] = k
    return pi


def select_target_cell_from_ancestor(ancestor_target, cells):
    """
    Given an ancestor target cell (a list of vertices) and the current partition (cells),
    return the first non-singleton cell in cells that is a subset of ancestor_target.
    """
    for cell in cells:
        if len(cell) > 1 and set(cell).issubset(set(ancestor_target)):
            return cell
    return None


def first_non_singleton(cells):
    """
    Return the first non-singleton cell in the given list of cells.
    """
    for cell in cells:
        if len(cell) > 1:
            return cell
    return None


def target_cell_select(node, cells):
    """
      - Starting from the immediate parent, check if the parent's target cell (if set)
        yields a non-singleton cell (in the current partition) that is a subset of it.
      - If found, return that cell immediately.
      - Otherwise, move to the parent's parent and repeat.
      - If no ancestor provides a candidate, return the first non-singleton cell in the
        overall partition.

    Returns:
      The selected target cell (a list of vertices), or None if no non-singleton cell exists.
    """
    ancestor = node.parent
    while ancestor is not None:
        if hasattr(ancestor, 'target_cell') and ancestor.target_cell is not None:
            candidate = select_target_cell_from_ancestor(ancestor.target_cell, cells)
            if candidate is not None:
                node.target_cell = candidate
                return candidate
        ancestor = ancestor.parent
    # Fallback: choose the first non-singleton cell in the overall partition
    candidate = first_non_singleton(cells)
    node.target_cell = candidate
    return candidate


'''Node Invariant/Traces, Graph Sorting Implementation'''

fuzz1 = [int("37541", 8), int("61532", 8), int("5257", 8), int("26416", 8)]
fuzz2 = [int("6532", 8), int("70236", 8), int("35523", 8), int("62437", 8)]


def FUZZ1(x):
    """Implements FUZZ1(x) = x XOR fuzz1[x & 3]."""
    return x ^ fuzz1[x & 3]


def MASHCOMM(l, i):
    """
    Commutative mix function.
    In Traces, defined as:  MASHCOMM(l, i) = (l + FUZZ1(i)).
    """
    return l + FUZZ1(i)


def MASHNONCOMM(l, i):
    """
    Non-commutative mix function.
    In Traces: MASHNONCOMM(l, i) = (FUZZ2(l) + i).
    """
    # Here we mix using fuzz2 on l.
    # Note: In our Python version we use l & 3 for consistency.
    return (fuzz2[l & 3] + i)


def MASH(l, i):
    """
    Alternate mix function.
    In Traces: MASH(l, i) = (((l ^ 065435) + i) & 077777).
    065435 (octal) -> int("65435", 8) and 077777 (octal) -> int("77777", 8).
    """
    return (((l ^ int("65435", 8)) + i) & int("77777", 8))


def MASH1(l, i):
    """
    Another mix function.
    In Traces: MASH1(l, i) = ((l + (i*i)) & 077777).
    """
    return ((l + i * i) & int("77777", 8))


def CLEANUP(l):
    """
    Cleanup function: in Traces defined as CLEANUP(l) = ((l) % 0x7FFF).
    """
    return l % 0x7FFF


# --- Invariant computation based on the Traces approach ---
def node_invariant(G, pi, sequence):
    """
    Computes an invariant value in the style of Traces.

    The invariant is built only from the partition (coloring) and the branch (individualization sequence).

    In Traces.c, the candidate structure accumulates a "singcode" by applying MASHCOMM
    for each singleton cell. Then the branch information is mixed in—using
    the color values.

    The final value is then cleaned-up (taken modulo 0x7FFF).
    """
    inv = 0
    n = len(pi)
    counts = Counter(pi)

    # Mix in every vertex that forms a singleton cell
    for v in range(n):
        if counts[pi[v]] == 1:
            inv = MASHCOMM(inv, v)

    # Mix in the sequence information.
    for v in sequence:
        color_value = pi[v]
        inv = MASHCOMM(inv, color_value)

    return CLEANUP(inv)


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
