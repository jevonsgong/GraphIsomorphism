import torch
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GCNConv, SAGPooling, global_mean_pool, MessagePassing
from torch_geometric.utils import softmax


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.SyncBatchNorm(hidden_channels)  # Batch normalization after fc1
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.bn2 = nn.SyncBatchNorm(out_channels)  # Batch normalization after fc2
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply BatchNorm after each linear layer
        x = self.relu(self.bn1(self.fc1(x)))  # Batch normalization after fc1 and ReLU
        x = self.bn2(self.fc2(x))  # Batch normalization after fc2
        return x


class GINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, num_layers):
        super(GINModel, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # BatchNorm layers for each GINConv

        # First GINConv layer
        mlp = MLP(in_channels, hidden_channels_list, hidden_channels_list)
        self.gin_layers.append(GINConv(mlp))
        self.bn_layers.append(nn.SyncBatchNorm(hidden_channels_list))  # BatchNorm after first GINConv

        # Intermediate GINConv layers
        for i in range(1, num_layers - 1):
            mlp = MLP(hidden_channels_list, hidden_channels_list, hidden_channels_list)
            self.gin_layers.append(GINConv(mlp))
            self.bn_layers.append(nn.SyncBatchNorm(hidden_channels_list))  # BatchNorm after each intermediate GINConv

        # Final GINConv layer
        mlp = MLP(hidden_channels_list, hidden_channels_list, hidden_channels_list)
        self.gin_layers.append(GINConv(mlp))
        self.bn_layers.append(nn.SyncBatchNorm(hidden_channels_list))
        #self.fc = nn.Linear(hidden_channels_list, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            if i < self.num_layers-1:
                #x = self.bn_layers[i](x)
                x = F.relu(x)
        #x = self.fc(x)
        return x


class GatedGINModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, num_layers):
        super(GatedGINModel, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # BatchNorm layers for each GINConv

        # First GINConv layer
        mlp = MLP(in_channels, hidden_channels_list, hidden_channels_list)
        self.gin_layers.append(GINConv(mlp))
        self.bn_layers.append(nn.SyncBatchNorm(hidden_channels_list))  # BatchNorm after first GINConv

        # Intermediate GINConv layers
        for i in range(1, num_layers - 1):
            mlp = MLP(hidden_channels_list, hidden_channels_list, hidden_channels_list)
            self.gin_layers.append(GINConv(mlp))
            self.bn_layers.append(nn.SyncBatchNorm(hidden_channels_list))  # BatchNorm after each intermediate GINConv

        # Final GINConv layer
        mlp = MLP(hidden_channels_list, hidden_channels_list, hidden_channels_list)
        self.gin_layers.append(GINConv(mlp))
        self.fc = nn.Linear(hidden_channels_list, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.bn_layers[i](x)  # Apply BatchNorm after each GINConv layer
            x = F.relu(x)
        x = self.fc(x)
        return x


class GatedGINLayer(nn.Module):
    """GINConv with a learnable gate. If gate≈0 the layer is skipped."""

    def __init__(self, in_ch, out_ch, alpha=None):
        super().__init__()
        self.mplayer = GINConv(nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch)))
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = nn.Parameter(torch.zeros(1))  # initial gate ~0.5

    def forward(self, x, edge_index):
        gate = torch.sigmoid(self.alpha)
        y = self.mplayer(x, edge_index)
        return gate * y, gate  # + (1-gate) * x


class Cand:
    __slots__ = ('col', 'x', 'trace', 'seq')

    def __init__(self, col, x, trace, seq):
        self.col = col  # Long[n]   current colors
        self.x = x  # (n,F)     node feat incl. colors
        self.trace = trace  # (d_trace) trace embedding
        self.seq = seq  # list[int] stabiliser sequence


class TracePool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sig):  # sig : (|X|, C)
        return sig.mean(0)


class BFS_Refine(nn.Module):
    def __init__(self, d_raw, d_hid, d_trace, k_layers):
        super().__init__()
        self.layers = nn.ModuleList([GatedGINLayer(d_raw + 1 + i, d_hid)
                                     for i in range(k_layers)])
        self.trace_pool = TracePool()  # 1 colour at start
        self.d_trace = d_trace

    # -- splitter : first non-singleton vertex
    @staticmethod
    def splitter(colours):
        cid = torch.unique(colours)[torch.bincount(colours).gt(1)][0]
        return int((colours == cid).nonzero(as_tuple=False)[0])

    # -- add one-hot for verts
    @staticmethod
    def add_channel(x, verts, n):
        ch = torch.zeros(n, 1, device=x.device);
        ch[verts] = 1.0
        return torch.cat([x, ch], 1)

    @staticmethod
    def signature(adj_t, verts, C, colors):
        oh = F.one_hot(colors, C).float()
        counts = torch.sparse.mm(adj_t, oh)  # (n,C)
        return counts[verts]  # (|X|,C)

    # ------------------------------------------------------------------
    def forward(self, data: Data):
        n = data.x.shape[0]
        col = torch.zeros(n, dtype=torch.long, device=data.x.device)
        x = torch.cat([data.x, col.unsqueeze(1).float()], 1)
        trace_emb = torch.zeros(64, device=x.device)

        gates = []
        layer_cands = [Cand(col, x, trace_emb, [])]
        for layer in self.layers:
            nxt = []
            for cand in layer_cands:
                h, g_scalar = layer(cand.x, data.edge_index)
                gates.append(g_scalar)

                # splitter vertex
                w = self.splitter(cand.col)
                cid = cand.col[w]
                verts = (cand.col == cid).nonzero(as_tuple=False).view(-1)

                x_new = self.add_channel(h, verts, n)
                col_new = cand.col.clone()
                col_new[verts] = col_new.max() + 1

                sig = self.signature(data.adj_t, verts, col_new.max() + 1, col_new)
                tr = (cand.trace + self.trace_pool(sig)) / 2

                nxt.append(Cand(col_new, x_new, tr, cand.seq + [w]))
            layer_cands = nxt  # BFS width =1 (exact path)

        out_trace = layer_cands[0].trace
        return out_trace, torch.stack(gates)  # (d_trace,), (L,)


class SimPLELoss(nn.Module):
    def __init__(self, r: float = 0.5, alpha: float = 0.5, b: float = -0.99, b_theta: float = 0.003, eps: float = 1e-8):
        super().__init__()
        self.r = r
        self.alpha = alpha
        # trainable bias terms (optional ‑ can be frozen if desired)
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float))
        self.b_theta = nn.Parameter(torch.tensor(b_theta, dtype=torch.float))
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, same: torch.Tensor):
        # cosine angle & magnitudes
        inner = (z1 * z2).sum(dim=-1)
        norm1 = z1.norm(dim=-1)
        norm2 = z2.norm(dim=-1)
        cos = inner / (norm1 * norm2 + self.eps)
        # Generalised inner‑product similarity with angular bias (Eq. 6)
        S = norm1 * norm2 * (cos - self.b_theta)
        # Loss components (Eq. 8)
        pos_term = self.alpha * same.float() * torch.log1p(torch.exp(-(S + self.b) / self.r))
        neg_term = (1.0 - self.alpha) * (~same).float() * torch.log1p(torch.exp(self.r * (S + self.b)))
        return (pos_term + neg_term).mean()

@torch.no_grad()
def momentum_update(model_q: nn.Module, model_k: nn.Module, m: float = 0.999):
    """MoCo‑style momentum update (θ_k ← m·θ_k + (1‑m)·θ_q)."""
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)

class GCN_RNI(nn.Module):
    """
    • Concatenates a fresh random vector z_v ∼ 𝒩(0,1)ᴰ to every node feature
      at *every* forward pass (exactly as in Algorithm 1 of the paper).
    • If your graphs are unlabeled, set `in_dim = 0` and it will work fine.
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 rni_dim: int    = 32,     # D in the paper
                 trainable_rni: bool = False):
        super().__init__()
        self.rni_dim = rni_dim
        self.trainable_rni = trainable_rni

        self.convs = nn.ModuleList()
        last_dim = in_dim + rni_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(last_dim, hidden_dim))
            last_dim = hidden_dim

        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

        if trainable_rni:
            # one learnable μ,σ per node dimension (optional variant)
            self.mu_rni  = nn.Parameter(torch.zeros(rni_dim))
            self.sig_rni = nn.Parameter(torch.ones (rni_dim))

    def _sample_rni(self, num_nodes, device):
        if self.trainable_rni:
            return self.mu_rni + self.sig_rni * torch.randn(num_nodes,
                                                            self.rni_dim,
                                                            device=device)
        return torch.randn(num_nodes, self.rni_dim, device=device)

    def forward(self, x, edge_index):
        # x may be None – handle unlabeled graphs
        if x is None:
            num_nodes = edge_index.max().item() + 1
            x = torch.empty(num_nodes, 0, device=edge_index.device)

        rni = self._sample_rni(x.size(0), x.device)
        h   = torch.cat([x, rni], dim=-1)      # (N, F+rni_dim)

        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        h = self.lin_out(h)
        return h                # graph embedding (B, out_dim)


class GCN_Pooling(nn.Module):
    """
    A lightweight 2-level pooling GNN:
      GCN → SAGPool → GCN → global_mean_pool.
    This mimics the hierarchical pooling setup used in the
    DiffPool paper but uses the PyG-native SAGPooling layer
    (memory-friendlier than DenseDiffPool).
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 pool_ratio: float = 0.5):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.pool = SAGPooling(hidden_dim, ratio=pool_ratio)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin_out  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.gcn1(x, edge_index))   # (N, hidden)
        h, edge_index, _, batch, _, _ = self.pool(h, edge_index, batch=batch)
        h = F.relu(self.gcn2(h, edge_index))
        h = self.lin_out(h)
        return h, batch

class GMNLayer(MessagePassing):
    """One cross-graph message-passing layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")              # node-level aggregation
        self.f_node  = nn.Linear(in_dim,  out_dim, bias=False)
        self.f_cross = nn.Linear(in_dim,  out_dim, bias=False)
        self.f_self  = nn.Linear(in_dim,  out_dim, bias=True)

    def forward(self, x_s, edge_index_s, x_t, edge_index_t, batch_s, batch_t):
        # 1) intra-graph update  (standard message passing)
        h_s = self.propagate(edge_index_s, x=x_s) + self.f_self(x_s)
        h_t = self.propagate(edge_index_t, x=x_t) + self.f_self(x_t)

        # 2) cross-graph attention
        a = torch.matmul(self.f_cross(x_s), self.f_node(x_t).T)  # [N_s, N_t]
        a = torch.softmax(a, dim=1)  # row-softmax

        x_s = F.relu(h_s + a @ x_t)  # (N_s, d)
        aT = torch.softmax(a.T, dim=1)  # (N_t, N_s)
        x_t = F.relu(h_t + aT @ x_s)  # (N_t, d)                 # ϕ_t
        return x_s, x_t

    def message(self, x_j):
        return x_j


class GraphMatchingNetwork(nn.Module):
    """
    • K cross-graph MP layers  (default K=3)
    • Graph embedding = global mean of final node states
    • Returns (B, out_dim) per graph
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int    = 256,
                 num_layers: int = 3):
        super().__init__()
        self.lin_in  = nn.Linear(in_dim, hidden_dim)
        self.layers  = nn.ModuleList([
            GMNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, g1, g2):
        """
        g1, g2 are torch_geometric.data.Batch objects.
        Returns pair of graph embeddings (B, out_dim).
        """
        x_s, edge_s, batch_s = g1.x, g1.edge_index, g1.batch
        x_t, edge_t, batch_t = g2.x, g2.edge_index, g2.batch

        x_s = F.relu(self.lin_in(x_s))
        x_t = F.relu(self.lin_in(x_t))

        for layer in self.layers:
            x_s, x_t = layer(x_s, edge_s, x_t, edge_t, batch_s, batch_t)

        return x_s, x_t, batch_s, batch_t