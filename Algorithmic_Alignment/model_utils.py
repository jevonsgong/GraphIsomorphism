import torch, numpy as np, scipy.sparse as sp
from torch import nn
from Exact_Algorithm import test
from Exact_Algorithm.main import R
import networkx as nx


class SignatureEmbed(nn.Module):
    def __init__(self, max_cells, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(max_cells, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim,  out_dim))

    def forward(self, sig):        # sig : [k] (padded to max_cells)
        return self.fc(sig.float())


class TraceCollector(nn.Module):
    def __init__(self,
                 WLStep,
                 max_parent: int,
                 max_cnt:    int,
                 max_cells:  int,
                 hid:        int = 32,
                 out_dim:    int = 64):
        super().__init__()
        self.n = max_cells
        self.out_dim = out_dim
        self.WLStep = WLStep
        self.emb_parent = nn.Embedding(max_parent + 1, hid)
        self.emb_cnt    = nn.Embedding(max_cnt    + 1, hid)
        self.sig_mlp    = nn.Sequential(
            nn.Linear(max_cells, hid),
            nn.ReLU(),
            nn.Linear(hid, hid))
        self.out_lin    = nn.Linear(2 * hid, out_dim)

    @torch.no_grad()
    def cell_embed(self, adj, X, cells, parent_colour, cnts_W):
        """
        adj            : torch sparse COO (n×n)
        X              : list[int]  vertices of current cell
        cells          : list[list[int]]  partition P_pi
        parent_colour  : int
        cnts_W         : LongTensor [|X|]   neighbour counts to splitter
        """
        # ---- (parentColour, count_W) component --------------------------
        pc_emb  = self.emb_parent(
                     parent_colour.clone())
        cnt_emb = self.emb_cnt(cnts_W).sum(0)          # sum-pool

        # ---- cell–edge signature component ------------------------------
        sig_vec = torch.zeros(self.n,
                              dtype=torch.int64,
                              device=adj.device)
        sig = self.WLStep.cell_edge_signature(adj, X, cells)       # tuple of tuples
        for vec in sig:                                # vec length = k
            sig_vec[:len(vec)] = torch.tensor(vec, device=adj.device)
            break                                      # all rows equal len(cells)

        #print(sig_vec)
        #print(sig_vec.shape)
        sig_emb = self.sig_mlp(sig_vec.float())

        # ---- fuse & project ---------------------------------------------
        z = torch.cat([pc_emb + cnt_emb, sig_emb], dim=-1)
        return self.out_lin(z)                         # [out_dim]



class WLStep(nn.Module):
    def __init__(self, device, n=512, *args, **kwargs):  # n is max_node of input_graphs
        super().__init__(*args, **kwargs)
        self.device = device
        self.tracecollector = TraceCollector(self, n, n, n)

    def cell_edge_signature(self, adj_coo, X, cells):
        """
        Parameters
        ----------
        adj_coo : torch.sparse_coo_tensor   shape (n,n)  unweighted graph
        X       : 1-D LongTensor | list[int]   vertices of the current cell
        cells   : list[list[int]]             partition P_π  (ordered)

        Returns
        -------
        signature : tuple[tuple[int]]
            Sorted tuple of degree-vectors, exactly as Traces expects.
        """
        device = adj_coo.device
        n = adj_coo.size(0)
        num_cell = len(cells)

        # 1.  Map every vertex v to its cell-index
        cell_id = torch.full((n,), -1, dtype=torch.long, device=device)
        for idx, cell in enumerate(cells):
            cell_id[torch.as_tensor(cell, dtype=torch.long, device=device)] = idx

        # 2.  Build a lookup  pos[v] = row-index in X  (or −1 if v∉X)
        X = torch.as_tensor(X, dtype=torch.long, device=device)
        X_sorted, _ = torch.sort(X)  # keep deterministic
        pos = torch.full((n,), -1, dtype=torch.long, device=device)
        pos[X_sorted] = torch.arange(X_sorted.numel(), device=device)

        # 3.  Gather all edges whose tail lies in X
        row, col = adj_coo.coalesce().indices()  # 2×nnz
        tail_mask = pos[row] >= 0  # tail in X
        src = pos[row[tail_mask]]  # 0 … |X|-1
        dst_cell = cell_id[col[tail_mask]]

        valid = dst_cell >= 0  # keep only mapped heads
        src = src[valid]
        dst_cell = dst_cell[valid]

        # 4.  Scatter-add to obtain counts[|X|, k]
        flat_idx = src * num_cell + dst_cell
        counts = torch.zeros(X_sorted.numel() * num_cell,
                             dtype=torch.int64, device=device)
        counts.scatter_add_(0, flat_idx,
                            torch.ones_like(flat_idx, dtype=torch.int64))
        counts = counts.view(X_sorted.numel(), num_cell)  # [|X|, k]

        # 5.  Convert to the exact signature format
        deg_vectors = [tuple(row.tolist()) for row in counts]  # per vertex
        signature = tuple(sorted(deg_vectors))  # lexicographic

        return signature

    def classify_by_edges_np(self, adj, X, W_mask):
        counts = adj@(W_mask.astype(np.int8))
        counts = [counts[i] for i in X]  # length |X|
        bucket = {}
        for v, c in zip(X, counts):
            bucket.setdefault(c, []).append(v)
        return [bucket[k] for k in sorted(bucket.keys())], torch.tensor(counts)

    def refinement(self, colors: torch.Tensor, adj, cells=None, alpha=None):
        adj_coo = self.csr_to_coo(adj)
        trace_stack = []
        n = colors.size(0)
        if not cells:
            cells = test.find_cells(colors.tolist())
        if not alpha:
            alpha = cells.copy()
        W = alpha.pop()
        W_mask = np.zeros(n, dtype=bool)
        W_mask[W] = True

        for X in cells:
            groups, counts = self.classify_by_edges_np(adj, X, W_mask)
            if len(groups) > 1:  # X is not split
                test.replace_cell(cells, X, groups)
                parent_col = colors[X[0]]
                t_emb = self.tracecollector.cell_embed(adj_coo, X, cells,
                                                   parent_col, counts)
                trace_stack.append(t_emb)
            #print("model groups", groups)
            if X in alpha:
                test.replace_cell(alpha, X, groups)
            else:
                test.append_largest_except_one(alpha, groups)
        new_color = test.find_color(cells)
        if trace_stack:
            trace = torch.stack(trace_stack).sum(0)
        else:
            trace = torch.zeros(self.tracecollector.out_dim)

        #print("trace", trace)
        return torch.tensor(new_color), cells, alpha, trace

    def Individualize(self, color, w):
        new_color = torch.clone(color)
        for v in range(len(color)):
            if color[v] < color[w] or v == w:
                continue
            else:
                new_color[v] = color[v] + 1
        return new_color

    def IterateWL(self, color, adj, splitter):
        n = len(color)
        new_color, last_color = color, torch.tensor(-1)
        new_color = self.Individualize(new_color, splitter)
        alpha = [splitter]
        new_cells = None
        trace_stack = []
        while max(new_color.tolist()) != n-1 and alpha:
            new_color, new_cells, alpha, cur_trace = self.refinement(new_color, adj, cells=new_cells, alpha=alpha)
            if cur_trace is not None:
                trace_stack.append(cur_trace)
        if trace_stack:
            trace = torch.stack(trace_stack)
            #print(trace.shape)
        else:
            trace = torch.zeros((1,self.tracecollector.out_dim))
        return new_color, trace.mean(0)

    def csr_to_coo(self, adj_csr):
        assert sp.issparse(adj_csr)
        indices = torch.tensor(
            np.array([adj_csr.nonzero()[0], adj_csr.nonzero()[1]]),
            dtype=torch.long)
        values = torch.ones(indices.size(1),
                            dtype=torch.float32,
                            )
        adj_coo = torch.sparse_coo_tensor(indices, values,
                                          (adj_csr.shape[0],
                                           adj_csr.shape[1]))
        return adj_coo

    def forward(self, colors_in, adj_csr, splitter=None):
        """
        colors_in : LongTensor [n]
        adj_csr : scipy.sparse.csr_matrix  (n × n)
        splitter : splitter vertices

        returns
        colors: Tensor [n]
        trace: Tensor [I,out_dim]
        """

        colors = colors_in.clone()

        if not splitter:
            colors, _, _, trace = self.refinement(colors, adj_csr)
            return colors, trace

        else:
            colors, trace = self.IterateWL(colors, adj_csr, splitter)

        return colors, trace




if __name__ == "__main__":
    device = torch.cuda.device("cuda:0") if torch.cuda.is_available() else "cpu"
    test_case_generator = test.generate_infinite_test_cases(10,15)
    WLRefine = WLStep(device)
    amount = 1
    for i in range(amount):
        G, G_iso, G_noniso = next(test_case_generator)
        pi_init = test.color_init(G)
        adj = nx.adjacency_matrix(G)
        pi_0 = test.refinement(G, pi_init, test.find_cells(pi_init))[0]
        n = G.number_of_nodes()
        pi_0_m = WLRefine.refinement(torch.tensor(pi_init), adj)[0]
        print("ini color", pi_0)
        print("ini model color", pi_0_m)
        assert pi_0 == pi_0_m.tolist()
        if max(pi_0) < len(pi_0) - 1:
            cells = test.find_cells(pi_0)
            target_cell = test.first_non_singleton(cells)
            print("target cell: ", target_cell)
            for v in [target_cell[0]]:
                pi_1 = R(v, G, pi_0)
                pi_1_m = WLRefine.IterateWL(pi_0_m, adj, v)
                print("algo refined color", pi_1[0])
                print("model refined color", pi_1_m[0])
