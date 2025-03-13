from collections import deque
from utils import *


def R(prefix, G, pi, cells=None):
    """ General function for computing the refined coloring. """
    if not prefix:
        if pi:
            return refinement(G,pi,find_cells(G, pi))
        else:
            return refinement(G,pi,cells)
    return refinement(G,individualization(pi,prefix),[[prefix]])


def canonical_labeling(G):
    """ Computes the canonical labeling of a graph G(V, E). """
    V = list(G.nodes())
    Leaves = []

    root = TreeNode([])  # Root is an empty sequence
    NodeQueue = deque([root])

    pi_0 = color_init(G)
    pi_init = R(None, G, pi_0)
    root.lc = pi_0
    root.rc = pi_init

    while NodeQueue:
        cur_node = NodeQueue.popleft()
        cur_cells = find_cells(G,cur_node.rc)
        TC = target_cell_select(cur_cells)
        for v in TC:
            NextNode = TreeNode(cur_node.sequence+[v])
            cur_node.children.append(NextNode)
            NextNode.lc = cur_node.rc
            NextNode.rc = R(v,G,NextNode.lc,cells=cur_cells)
            NextNode.traces = N(G,NextNode.rc)
            if NextNode.rc == len(V):
                Leaves.append(NextNode)
            else:
                NodeQueue.append(NextNode)

    BestNode = max(Leaves, key=lambda node: node.traces)
    C_G = BestNode.rc

    return C_G

'''Testing Isomorphism'''
