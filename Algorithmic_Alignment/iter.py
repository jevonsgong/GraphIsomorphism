import hashlib

from Algorithmic_Alignment.model_utils import *
import torch
import collections
import networkx as nx
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

device = torch.cuda.device("cuda:0") if torch.cuda.is_available() else "cpu"

class CanonicalNet_nolearn(nn.Module):
    class Node:
        __slots__ = ("seq",           # list[int]  sequence of fixed vertices
                     "colors",        # torch.LongTensor [n]
                     "trace")         # torch.LongTensor [out_dim]

        def __init__(self, seq, colors, trace):
            self.seq = seq
            self.colors = colors
            self.trace = trace

        def __lt__(self, other):
            return (self.stable_hash(self.trace), self.seq) < (self.stable_hash(other.trace), other.seq)

        @staticmethod
        def stable_hash(arr: torch.Tensor) -> int:
            """SHA1 hash of a 1-D LongTensor → 64-bit int."""
            h = hashlib.sha1(arr.cpu().numpy().tobytes()).digest()[:8]
            return int.from_bytes(h, "little")

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.refiner = WLStep(self.device)

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

    def canonical_colors(self, G):
        """
        Return the canonical color vector
        """
        n = nx.number_of_nodes(G)
        A_csr = nx.adjacency_matrix(G)
        A_csr = A_csr + A_csr.T
        A_csr[A_csr > 1] = 1

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

            for v in tgt:
                if v in node.seq:
                    continue

                seq_child = node.seq + [int(v)]
                colors_child, trace_step = self.refiner(
                    node.colors.clone(), A_csr, v)
                trace_child = trace_step

                child = self.Node(seq_child, colors_child, trace_child)

                if self.is_discrete(colors_child):
                    leaves.append(child)
                else:
                    queue.append(child)

        best = max(leaves)            # uses __lt__ defined above
        return best.colors

    # --------------- convenience wrapper ---------------------------------
    def canonical_graph(self, G) -> nx.Graph:
        """
        Return a *relabelled copy* of G whose node ordering is canonical.
        """
        colours = self.canonical_colors(G)
        mapping = {old: int(colours[i].item())
                   for i, old in enumerate(G.nodes())}
        return nx.relabel_nodes(G, mapping, copy=True)

    def forward(self, x):
        return self.canonical_graph(x)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    n = 15
    G1 = nx.random_regular_graph(12, n)
    perm = list(range(n)); random.shuffle(perm)
    G2 = nx.relabel_nodes(G1, {i: perm[i] for i in range(n)})

    canoniser = CanonicalNet_nolearn()
    C1 = canoniser.canonical_graph(G1)
    C2 = canoniser.canonical_graph(G2)
    print(C1.edges)
    print(C2.edges)

    print("Canonical equal? ", canonical_representation(C1)==canonical_representation(C2))  # → True

