import torch
from torch import nn
from Exact_Algorithm.test import *
from Exact_Algorithm.utils import *

device = torch.cuda.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class WLStep(nn.Module):
    def stable_hash(self, int_tensor):
        # convert to bytes
        b = int_tensor.numpy().tobytes()
        digest = hashlib.blake2b(b, digest_size=8).digest()  # 64-bit
        return torch.tensor(int.from_bytes(digest, 'little'), dtype=torch.int64)

    def rank_unique(self, int_ids):
        uniq, inv = torch.unique(int_ids, sorted=True, return_inverse=True)
        return inv.to(torch.int64)

    def forward(self, color, adj):
        if not isinstance(color, torch.types.Tensor):
            color = torch.tensor(color)
        n = color.shape[0]
        k = color.max().item() + 1

        row, col = adj.indices()
        hist = torch.zeros((n, k), dtype=torch.int64)
        hist.index_put_((row, color[col]), torch.tensor(1), accumulate=True)

        sig = torch.cat([color.unsqueeze(1), hist], dim=1)
        hashed = self.stable_hash(sig)
        new_color = self.rank_unique(hashed)

        return new_color.to(torch.int64)

    def IterateWL(self, color, adj):
        new_color = torch.tensor(-1)
        i = 0
        while new_color != color:
            if i > adj.shape[0] - len(set(color)):
                print("WL Refinement Error!")
                return color  # unchanged if error
            new_color = self.forward(color, adj)
            i += 1
        return new_color

"""class CanonicalNet_NoLearn(nn.Module):
    def forward(self, G_adj, n):
        color = torch.zeros(n, dtype=torch.int64)
        code = torch.tensor(0, dtype=torch.int32)
        trace = []
        queue = deque(find_cells(color))  # α initial

        while max(color) < n - 1:
            # 1. refinement loop
            color = iterate_WL(color, G_adj, queue, trace, code)
            if max(color) == n - 1: break  # discrete → leaf

            # 2. choose target cell (heuristic)
            cells = partition_from_colours(color)
            C = pick_target_cell(cells)
            for v in C:
                new_colours = individualization(color, v)
                # push to search queue (BFS order like Traces)
                queue.append((new_colours, [v], trace, code))
            # pop next state
            color, seq, trace, code = queue.popleft()

        return canonical_label_from_colours(color)"""


if __name__ == "__main__":
    test_case_generator = generate_infinite_test_cases()
    WLRefine = WLStep()
    amount = 10
    for i in range(amount):
        G, G_iso, G_noniso = next(test_case_generator)
        pi_init = color_init(G)
        adj = nx.adjacency_matrix(G)
        pi_0 = refinement(G,pi_init,find_cells(pi_init))
        pi_0_WL = WLRefine.IterateWL(pi_init, adj)
        print(pi_0)
        print(pi_0_WL)