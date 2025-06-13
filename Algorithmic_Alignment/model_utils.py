import torch, numpy as np, scipy.sparse as sp
from torch import nn
from Exact_Algorithm.test import *
from Exact_Algorithm.utils import *


class TracesHead(torch.nn.Module):

    def __init__(self, color_refiner):
        super().__init__()
        self.refiner = color_refiner

    @staticmethod
    def stable_hash(arr: torch.Tensor) -> int:
        """SHA1 hash of a 1-D LongTensor → 64-bit int."""
        h = hashlib.sha1(arr.cpu().numpy().tobytes()).digest()[:8]
        return int.from_bytes(h, "little")

    def edge_signature(self, adj_coo, cell_mask, n_cells, colors):
        # adjacency-vector of the cell
        v_mask = cell_mask.to(torch.float32).unsqueeze(1)  # (n,1)
        neigh = torch.sparse.mm(adj_coo, v_mask).view(-1).to(torch.int64)  # (n,)
        # aggregate counts per colour
        sig = torch.zeros(n_cells, dtype=torch.long,
                          device=colors.device)
        #print(colors.dtype)
        #print(neigh.dtype)
        sig.scatter_add_(0, colors, neigh)
        return sig

    def forward(self, colors_in, adj_csr, alpha=None):
        """
        colors_in : LongTensor [n]      initial colours
        adj_csr : scipy.sparse.csr_matrix  (n × n)
        alpha : splitter vertices
        """

        assert sp.issparse(adj_csr)
        indices = torch.tensor(
            np.array([adj_csr.nonzero()[0], adj_csr.nonzero()[1]]),
            dtype=torch.long, device=colors_in.device)
        values = torch.ones(indices.size(1),
                            dtype=torch.float32,
                            device=colors_in.device)
        adj_coo = torch.sparse_coo_tensor(indices, values,
                                          (adj_csr.shape[0],
                                           adj_csr.shape[1]))

        colors = colors_in.clone()
        trace = []

        if not alpha:
            colors = self.refiner.global_refinement(colors, adj_csr)
            trace = (0,0,0)
            return colors, trace

        for w in alpha:
            cell_id = colors[w].item()
            cell_mask = (colors == cell_id)
            bucket_sz = cell_mask.sum().item()

            n_cells = int(colors.max().item() + 1)
            edge_sig = self.edge_signature(adj_coo,
                                           cell_mask,
                                           n_cells,
                                           colors)  # (n_cells,)

            trace.append((cell_id,
                          bucket_sz,
                          self.stable_hash(edge_sig)))

            colors = self.refiner.IterateWL(colors, adj_csr, w)

        return colors, tuple(trace)


class WLStep(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def record_trace(self, X, colors, groups):
        sizes = tuple(len(g) for g in groups)
        return (colors[X[0]], sizes,)

    def scipy_to_torch_coo(self, spmat):
        if not sp.issparse(spmat):
            raise TypeError("Input must be a SciPy sparse")

        coo = spmat.tocoo(copy=False)
        # 2×nnz index matrix
        idx = np.vstack((coo.row, coo.col)).astype(np.int64)
        # convert to torch
        indices = torch.from_numpy(idx).to(self.device)
        values = torch.from_numpy(coo.data).to(self.device, dtype=torch.float32)
        shape = torch.Size(coo.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def classify_by_edges_np(self, adj, X, W_mask):
        counts = adj[X].dot(W_mask.astype(np.int8))  # length |X|
        bucket = {}
        for v, c in zip(X, counts):
            bucket.setdefault(c, []).append(v)

        return [bucket[k] for k in sorted(bucket.keys())]

    def global_refinement(self, colors, adj):
        n = colors.size(0)
        cells = [list(range(n))]
        alpha = [cells[0]]

        while alpha and len(cells) < n:
            W = alpha.pop()
            W_mask = np.zeros(n, dtype=bool)
            W_mask[W] = True

            i = 0
            while i < len(cells):
                X = cells[i]
                groups = self.classify_by_edges_np(adj, X, W_mask)

                if len(groups) == 1:  # X is not split
                    i += 1
                    continue

                cells[i:i + 1] = groups

                if X in alpha:
                    j = alpha.index(X)
                    alpha.pop(j)
                    for g in groups[::-1]:
                        alpha.insert(j, g)
                else:
                    largest = max(groups, key=len)
                    for g in groups:
                        if g is not largest:
                            alpha.append(g)

                i += len(groups)

        color_np = np.empty(n, dtype=np.int64)
        for idx, cell in enumerate(cells):
            color_np[cell] = idx
        return torch.from_numpy(color_np)

    def refinement(self, colors, adj, alpha):
        device = colors.device
        n = colors.size(0)
        cells = find_cells(list(colors.detach().cpu()))
        W = alpha.pop()
        W_mask = np.zeros(n, dtype=bool)
        W_mask[W] = True

        i = 0
        while i < len(cells):
            X = cells[i]
            groups = self.classify_by_edges_np(adj, X, W_mask)

            if len(groups) == 1:  # X is not split
                i += 1
                continue

            cells[i:i + 1] = groups

            if X in alpha:
                j = alpha.index(X)
                alpha.pop(j)
                for g in groups[::-1]:
                    alpha.insert(j, g)
            else:
                largest = max(groups, key=len)
                for g in groups:
                    if g is not largest:
                        alpha.append(g)

            i += len(groups)

        color_np = np.empty(n, dtype=np.int64)
        for idx, cell in enumerate(cells):
            color_np[cell] = idx
        return torch.from_numpy(color_np), alpha

    def Individualize(self, color, w):
        new_color = torch.clone(color)
        for v in range(len(color)):
            if color[v] < color[w] or v == w:
                continue
            else:
                new_color[v] = color[v] + 1
        return new_color

    def IterateWL(self, color, adj, splitter):
        new_color, last_color = color, torch.tensor(-1)
        i = 0
        new_color = self.Individualize(new_color, splitter)
        alpha = [splitter]
        while not torch.equal(new_color, last_color) and alpha:
            temp = new_color
            new_color, alpha = self.refinement(new_color, adj, alpha)
            # print("new color: ", new_color)
            last_color = temp
            i += 1
        return new_color



if __name__ == "__main__":
    device = torch.cuda.device("cuda:0") if torch.cuda.is_available() else "cpu"
    test_case_generator = generate_infinite_test_cases()
    WLRefine = WLStep(device)
    amount = 1
    for i in range(amount):
        G, G_iso, G_noniso = next(test_case_generator)
        pi_init = color_init(G)
        adj = nx.adjacency_matrix(G)
        pi_0 = refinement(G, pi_init, find_cells(pi_init))[0]
        pi_0_m = WLRefine.global_refinement(torch.tensor(pi_init), adj)
        assert pi_0 == list(pi_0_m)
        print("ini color", pi_0)
        print("refined color", pi_0_m)
        if max(pi_0) < len(pi_0) - 1:
            cells = find_cells(pi_0)
            target_cell = first_non_singleton(cells)
            for v in target_cell:
                pi_1 = R(v, G, pi_0)
                pi_1_m = WLRefine.IterateWL(pi_0_m, adj, v)
                print("algo refined color", pi_1[0])
                print("model refined color", pi_1_m)
