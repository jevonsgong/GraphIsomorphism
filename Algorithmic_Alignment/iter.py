from Algorithmic_Alignment.model_utils import *
import torch
import collections
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

device = torch.cuda.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class MLPScore(nn.Module):
    """
    Score 1 vertex at a time.
    Input  feature  x_v = [deg(v), colour(v), is_fixed(v)]
           (all integers, we embed them and pass through 2-layer MLP)
    Output scalar score (the *lower* → explored earlier in BFS).
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.embed_deg   = nn.Embedding(  512,  8)   # clip degrees
        self.embed_col   = nn.Embedding( 1024,  8)   # clip colours
        self.embed_fixed = nn.Embedding(    2,  2)

        in_dim = 8 + 8 + 2
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    def forward(self, deg, col, fixed):
        z = torch.cat([ self.embed_deg(deg.clip(max=511)),
                        self.embed_col(col.clip(max=1023)),
                        self.embed_fixed(fixed) ], dim=-1)
        z = F.relu(self.lin1(z))
        return self.lin2(z).squeeze(-1)

class CanonicalNet_nolearn:
    class Node:
        __slots__ = ("seq",           # list[int]  sequence of fixed vertices
                     "colors",        # torch.LongTensor [n]
                     "trace")         # tuple[tuple]  cumulative trace

        def __init__(self, seq, colors, trace):
            self.seq = seq
            self.colors = colors
            self.trace = trace

        def __lt__(self, other):
            return (self.trace, self.seq) < (other.trace, other.seq)

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.refiner = TracesHead(WLStep(self.device)).to(self.device)
        self.scorer = MLPScore().to(self.device)

    @staticmethod
    def is_discrete(colors):
        return int(colors.max().item()) == colors.numel() - 1

    @staticmethod
    def first_non_singleton_cell(colors):
        colors_np = colors.cpu().numpy()
        _, inv, counts = np.unique(colors_np,
                                   return_inverse=True,
                                   return_counts=True)

        for cid, cnt in enumerate(counts):
            if cnt > 1:
                return np.where(inv == cid)[0].tolist()
        return []

    def canonical_colors(self, G: nx.Graph):
        """
        Return the canonical color vector
        """
        n = nx.number_of_nodes(G)
        A_csr = nx.adjacency_matrix(G)
        A_csr = A_csr + A_csr.T
        A_csr[A_csr > 1] = 1

        degs = torch.tensor(A_csr.sum(axis=1).A1,
                            dtype=torch.long,
                            device=self.device)

        colors_0 = torch.zeros(n, dtype=torch.long, device=self.device)
        colors_0, trace_init = self.refiner(colors_0, A_csr)
        root = self.Node([], colors_0, ())
        if self.is_discrete(colors_0):
            return colors_0            # trivial 1-vertex graph

        queue = collections.deque([root])
        leaves = []

        while queue:
            node = queue.popleft()

            tgt = self.first_non_singleton_cell(node.colors)
            if not tgt:               # already discrete
                leaves.append(node)
                continue

            fixed_flag = torch.zeros(n, dtype=torch.long, device=self.device)
            fixed_flag[node.seq] = 1

            with torch.no_grad():  # inference mode
                scores = self.scorer(degs[tgt],
                                     node.colors[tgt],
                                     fixed_flag[tgt])
            order = [tgt[i] for i in scores.argsort()]

            for v in order:  # explore all → exactness kept
                seq_child = node.seq + [int(v)]
                col_child, tr_step = self.refiner(
                    node.colors.clone(), A_csr, alpha_queue=[v])
                trace_child = node.trace + tr_step
                child = self.Node(seq_child, col_child, trace_child)

                if self.is_discrete(col_child):
                    leaves.append(child)
                else:
                    queue.append(child)

        best = max(leaves)            # uses __lt__ defined above
        return best.colors

    # --------------- convenience wrapper ---------------------------------
    def canonical_graph(self, G: nx.Graph) -> nx.Graph:
        """
        Return a *relabelled copy* of G whose node ordering is canonical.
        """
        colours = self.canonical_colors(G)
        mapping = {old: int(colours[i].item())
                   for i, old in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping, copy=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    G1 = nx.cycle_graph(6)
    perm = list(range(6)); random.shuffle(perm)
    G2 = nx.relabel_nodes(G1, {i: perm[i] for i in range(6)})

    canoniser = CanonicalNet_nolearn()
    C1 = canoniser.canonical_graph(G1)
    C2 = canoniser.canonical_graph(G2)
    print("Canonical equal? ", nx.is_isomorphic(C1, C2))  # → True

