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


'''Individualization-Refinement Implementation'''


def color_init(G):
    """ Initialize coloring """
    pi = [0] * nx.number_of_nodes(G)
    return pi


def classify_by_edges(G, X, W):
    """ Classify vertices in X into groups based on the number of edges to W. """
    edge_count = defaultdict(list)

    for v in X:
        count = 0
        for w in W:
            if w in list(G.neighbors(v)):
                count += 1
        edge_count[count].append(v)

    return list(edge_count.values())


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
    '''
    pi_prime = pi.copy()
    for v in range(len(pi)):
        if pi[v] < pi[w] or v == w:
            continue
        else:
            pi_prime[v] = pi[v] + 1
    '''
    pi_prime = pi.copy()
    pi_prime[w] = max(pi) + 1  # Assign new color
    return pi_prime



def refinement(G, pi, alpha):
    """ Perform the Refinement step F(G, pi, alpha) """
    cells = find_cells(G, pi)
    while alpha and max(cells) != len(pi) - 1:
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
    """ Transform from color to cells """
    cells = [[] for i in range(len(set(pi)))]
    for node, color in enumerate(pi):
        cells[color].append(node)
    return cells


def find_color(G, cells):
    """ Transform from cells to color """
    pi = [0] * G.number_of_nodes()
    for k, cell in enumerate(cells):
        for v in cell:
            pi[v] = k
    return pi


def target_cell_select(cells):
    """ Target Cell Selector """
    return max(cells, key=len)


'''Node Invariant/Traces, Graph Sorting Implementation'''


def N(G, pi):
    """ Node Invariant function, a deterministic function that sorts partitions """
    return hash_graph(nx.quotient_graph(G, find_cells(G, pi)))


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

    print("example graph hashcode:", hash_graph(G))
    print("example Node Invariant value(quotient graph hashcode):", N(G, pi_2R))
