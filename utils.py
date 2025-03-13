from collections import deque,defaultdict
import random
import networkx as nx

class TreeNode:
    def __init__(self, sequence):
        self.sequence = sequence  # Sequence of graph nodes
        self.lc = None  # Last color
        self.rc = None  # Refined color
        self.traces = None  # Traces/Node Invariant
        self.children = []


def color_init(G):
    """Initialize coloring"""
    pi = [0]*nx.number_of_nodes(G)
    return pi

def classify_by_edges(G, X, W):
    """Classify vertices in X into groups based on the number of edges to W."""
    edge_count = defaultdict(list)

    for v in X:
        count = 0
        for w in W:
            if w in list(G.neighbors(v)):
                count+=1
        edge_count[count].append(v)

    return list(edge_count.values())


def replace_cell(partition, X, new_cells):
    """Replace a cell X in partition with new cells."""
    partition.remove(X)
    partition.extend(new_cells)
    return 0


def append_largest_except_one(alpha, new_cells): #checked
    """Append all but the largest cell in new_cells to alpha."""
    if not new_cells:
        return alpha

    largest_cell = max(new_cells, key=len)
    for cell in new_cells:
        if cell != largest_cell:
            alpha.append(cell)

    return alpha

def individualization(pi, TC):
    """ Perform the Individualization step I(pi, w) -> pi' """
    pi_prime = pi.copy()
    w = TC[0]  # pick the first vertex from target cell
    for v in range(len(pi)):
        if pi[v] < pi[w] or v == w:
            continue
        else:
            pi_prime[v] = pi[v] + 1
    return pi_prime, w


def refinement(G, pi, alpha):
    """ Perform the Refinement step F(G, pi, alpha) """
    cells = find_cells(G,pi)
    alpha_queue = deque(alpha)
    # print(alpha_queue)
    node_count = len(pi) # G.number_of_nodes()
    while alpha_queue and node_count!=len(cells):
        W = alpha_queue.popleft()
        # print(W)
        for X in cells:
            groups = classify_by_edges(G, X, W)
            replace_cell(cells, X, groups)

            if X in alpha_queue:
                replace_cell(alpha_queue, X, groups)
            else:
                append_largest_except_one(alpha_queue, groups)

    return find_color(G,cells)

def N(prefix, G, pi):
    """ Node Invariant function for computing traces. """
    Q = nx.quotient_graph(G, pi)
    return Q

def find_cells(G, pi):
    """Transform from color to cells"""
    cells = defaultdict(list)
    for i in range(G.number_of_nodes()):
        cells[pi[i]].append(i)
    return list(cells.values())

def find_color(G, cells):
    """Transform from cells to color"""
    pi = [0]*G.number_of_nodes()
    for k,cell in enumerate(cells):
        for v in cell:
            pi[v] = k
    return pi

def target_cell_select(cells):
    """Target Cell Selector"""
    return max(cells, key=len)

G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])

pi_0 = color_init(G)
print("example initial labeling:", [i for i in range(G.number_of_nodes())])
print("example initial color:", pi_0)
print("example cells:", find_cells(G,pi_0))
assert find_color(G,find_cells(G,pi_0)) == pi_0

pi_i = refinement(G,pi_0,find_cells(G,pi_0))
print("example initial refined color:", pi_i)

pi_1I, w = individualization(pi_i,target_cell_select(find_cells(G,pi_i)))
print("example first IR individualized color:", pi_1I)
pi_1R = refinement(G,pi_1I,[[w]])
print("example first IR refined color:", pi_1R)

pi_2I, w = individualization(pi_1R,target_cell_select(find_cells(G,pi_1R)))
print("example Second IR individualized color:", pi_2I)
pi_2R = refinement(G,pi_2I,[[w]])
print("example Second IR refined color:", pi_2R)
final_cell = find_cells(G,pi_2R)
print("example final cell:", final_cell)

print(N([],G,final_cell))

