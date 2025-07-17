import hashlib
import struct
from collections import defaultdict

import torch
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F
import xxhash
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GCNConv, SAGPooling, global_mean_pool, MessagePassing
from torch_geometric.utils import softmax, to_dense_batch, to_dense_adj


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
        # self.fc = nn.Linear(hidden_channels_list, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            if i < self.num_layers - 1:
                # x = self.bn_layers[i](x)
                x = F.relu(x)
        # x = self.fc(x)
        return x


class rGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, num_layers):
        super(rGIN, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()  # BatchNorm layers for each GINConv
        self.r_dim = 1

        # First GINConv layer
        mlp = MLP(in_channels + self.r_dim, hidden_channels_list, hidden_channels_list)
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
        # self.fc = nn.Linear(hidden_channels_list, out_channels)

    def forward(self, x, edge_index):
        r = torch.randint(100, size=(len(x), 1)).float() / 100
        x = torch.cat([x, r], 1).to(self.device)
        for i in range(self.num_layers):
            x = self.gin_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            if i < self.num_layers - 1:
                # x = self.bn_layers[i](x)
                x = F.relu(x)
        # x = self.fc(x)
        return x


class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(GCNConv(in_dim + self.rni_dim, hidden_dim))

        for i in range(1, num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        h = x
        for i in range(self.num_layers):
            h = self.gcn_layers[i](h, edge_index)
            if i < self.num_layers - 1:
                h = F.relu(h)
        # x = self.fc(x)

        return h


class GCN_RNI(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 random_ratio: float = 0.875,  # following GCN_RNI paper best setting
                 ):
        super().__init__()
        self.rni_dim = int(hidden_dim * random_ratio)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim - self.rni_dim)
        # self.bn_layers = nn.ModuleList()
        # self.mlp_layers = nn.ModuleList()
        # self.mlp = MLP(hidden_dim,hidden_dim,hidden_dim)

        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        for i in range(1, num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.activation = F.elu

    def forward(self, x, edge_index):
        random_dims = torch.empty(x.shape[0], self.rni_dim).to(x.device)  # Random INIT
        torch.nn.init.normal_(random_dims)
        h = self.activation(self.conv1(x, edge_index))
        h = self.activation(self.conv2(h, edge_index))
        h = torch.cat((h, random_dims), dim=1)
        for i in range(self.num_layers):
            h = self.gcn_layers[i](h, edge_index)
            # h = self.mlp_layers[i](h)
            # h = self.bn_layers[i](h)
            if i < self.num_layers - 1:
                h = self.activation(h)
        # x = self.fc(x)

        return h


class GCN_Pooling(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 pool_ratio: float = 0.5):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.pool = SAGPooling(hidden_dim, ratio=pool_ratio)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.gcn1(x, edge_index))  # (N, hidden)
        h, edge_index, _, batch, _, _ = self.pool(h, edge_index, batch=batch)
        h = F.relu(self.gcn2(h, edge_index))
        h = self.lin_out(h)
        return h, batch


class GMNLayer(MessagePassing):
    """One cross-graph message-passing layer."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__(aggr="add")  # node-level aggregation
        self.mlp = MLP(in_dim * 3, hidden * 3, out_dim)

    def forward(self, node_states, edges1, edges2, lengths):
        x1, x2 = [], []
        idx = 0
        for length in lengths:
            x1.append(node_states[idx: idx + length])
            x2.append(node_states[idx + length: idx + 2 * length])
            idx += 2 * length
        aggr_msg1 = self.propagate(edges1, x=torch.cat(x1, dim=0))
        aggr_msg2 = self.propagate(edges2, x=torch.cat(x2, dim=0))
        # print("Single msg", aggr_msg1.shape)
        aggr_msg = torch.stack((aggr_msg1, aggr_msg2), dim=2).view(-1, 256)
        # print("Combine msg", aggr_msg.shape)
        attention = []
        for x, x_ in zip(x1, x2):
            a = torch.mm(x, torch.transpose(x_, 1, 0))
            a_x = torch.softmax(a, dim=1)  # i->j
            a_y = torch.softmax(a, dim=0)  # j->i
            attention_x = torch.mm(a_x, x_)
            attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
            attention.append(attention_x)
            attention.append(attention_y)
        attention = torch.cat(attention, dim=0)
        # print("node_state", node_states.shape)
        # print("att", attention.shape)
        # print("agr", aggr_msg.shape)
        attention_input = node_states - attention
        node_state_inputs = torch.cat([aggr_msg, attention_input, node_states], dim=-1)
        out = self.mlp(node_state_inputs)
        return out

    def message(self, x_j):
        return x_j


class GMNModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GMNLayer(in_dim, hidden, out_dim))
            else:
                self.layers.append(GMNLayer(out_dim, hidden, out_dim))
        # self.out = nn.Linear(3*out_dim, out_dim)

    def forward(self, g1, g2):
        b1 = g1.to_data_list()
        b2 = g2.to_data_list()
        x1, x2 = [], []
        node_states = []
        edges1 = []
        edges2 = []
        lengths = []
        for d1, d2 in zip(b1, b2):
            x1.append(d1.x)
            x2.append(d2.x)
            node_states.append(d1.x)
            node_states.append(d2.x)
            edges1.append(d1.edge_index)
            edges2.append(d2.edge_index)
            lengths.append(d1.x.shape[0])
        node_states = torch.cat(node_states, dim=0)  # [N_nodes, 256]
        # print("Node states",node_states.shape)
        edges1 = torch.cat(edges1, dim=1)
        edges2 = torch.cat(edges2, dim=1)
        for layer in self.layers:
            node_states = layer(node_states, edges1, edges2, lengths)
        x1, x2 = [], []
        idx = 0
        for length in lengths:
            x1.append(node_states[idx: idx + length])
            x2.append(node_states[idx + length: idx + 2 * length])
            idx += 2 * length
        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        return x1, x2


class Cand:
    __slots__ = ('color', 'x', "gx", 'trace')

    def __init__(self, color, x, gx, trace):
        self.color = color  # current colors
        self.x = x  # (n,F)     node feat incl. colors
        self.gx = gx  # list of per graph node feat
        self.trace = trace
        # self.gate = gate
        # self.seq = seq  # list[int] stabiliser sequence


class BFS_Refine(nn.Module):
    def __init__(self, in_dim, hidden, num_layers, max_width, bs, max_nodes=256, world_size=4, one_hot=True):
        super().__init__()
        self.layers = nn.ModuleList()
        if max_nodes is None:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GatedGINLayer(in_dim + 1, hidden, is_root=True))
                else:
                    self.layers.append(GatedGINLayer(hidden + 1, hidden))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(GatedGINLayer(in_dim + max_nodes, hidden, is_root=True))
                else:
                    self.layers.append(GatedGINLayer(hidden + max_nodes, hidden))
        self.max_nodes = max_nodes
        # self.batch = None
        self.max_width = max_width
        self.bs = bs
        self.world_size = world_size
        self.one_hot = one_hot

    def get_colors_and_traces(self, x, batch, last_trace, v=None, Adjs=None):  # x shape [N, hidden]
        flat_x, node_nums = to_dense_batch(x, batch)
        colors = []
        traces = []
        g_xs = []
        # cells = []
        # graphs = []
        for i in range(node_nums.shape[0]):
            node_num = torch.sum(node_nums[i])
            g_x = flat_x[i, :node_num, :]  # [graph_node_num, hidden]
            colors.append(self.embeddings_to_color(g_x)[0])
            g_xs.append(g_x)
            if v is not None:
                if v[i] != -1:
                    hashes = self.embeddings_to_color(g_x)[1]
                    Adj = Adjs[i].to(x.device).to(torch.float32)
                    traces.append((Adj[v[i], :] @ hashes).item())
                else:
                    traces.append(last_trace[i])
            else:
                traces.append(0)
            # cells.append(self.embeddings_to_color(g_x)[1])
        return colors, torch.tensor(traces).to(x.device), g_xs

    def embeddings_to_color(self, H, q=10000, eps=0):  # decides how strict we treat two nodes to have the same color
        n, d = H.shape
        norms = torch.norm(H, dim=1, keepdim=True) + eps
        Hn = H / norms
        # print(H)

        Hq = torch.round(Hn * q).to(torch.float32)
        # print(Hq)
        hashes = torch.sum(Hq, dim=1)
        # print(hashes)
        buckets = defaultdict(list)
        for v, h in enumerate(hashes):
            buckets[h.item()].append(v)
        cells = list(buckets.values())

        cells.sort(key=lambda cell: min(cell))
        pi = torch.empty(n, dtype=torch.int64)
        for color_id, cell in enumerate(cells):
            for node in cell:
                pi[node] = color_id
        # print(pi)
        return pi, hashes

    # -- splitter : return first largest non-singleton cell
    @staticmethod
    def splitter(colors):
        cid = torch.unique(colors)[int(torch.argmax(torch.bincount(colors)))]
        return (colors == cid).nonzero().squeeze(-1)

    # -- add one-hot for verts
    @staticmethod
    def individualize(pi, w):
        pi_prime = pi.clone()
        for v in range(len(pi)):
            if pi[v] < pi[w] or v == w:
                continue
            else:
                pi_prime[v] = pi[v] + 1
        return pi_prime

    @staticmethod
    def xxhash_all_rows(x_int: torch.Tensor) -> torch.Tensor:
        assert x_int.dtype == torch.int64
        n, d = x_int.shape
        out = torch.zeros(n, dtype=torch.float64).to(x_int.device)
        np_view = x_int.cpu().numpy()  # zero-copy
        for i in range(n):
            digest_u64 = xxhash.xxh64_intdigest(np_view[i].tobytes())
            digest_i64 = struct.unpack('<q', struct.pack('<Q', digest_u64))[0]
            out[i] = digest_i64
            out[i] = digest_i64
        return out

    # ------------------------------------------------------------------
    def forward(self, data, Adjs):
        batch = data.batch  # Remember batch for graph segment
        device = data.x.device
        n = data.x.shape[0]  # total number of nodes in a batch
        color = torch.zeros((n, 1), device=device).int()
        if not self.one_hot:
            x = torch.cat([data.x, color / self.max_nodes], 1)  # x.shape = [n, F_dim+1], max_nodes=256
        else:
            one_hot_color = F.one_hot(color.long(), num_classes=self.max_nodes).int().squeeze(1)
            x = torch.cat([data.x, one_hot_color], 1)
        # x = data.x
        gates = []
        trace = torch.zeros([self.bs // self.world_size,1])
        is_root = True
        layer_cands = [Cand(color, x, None, trace)]
        end_flag = False
        depth = 0
        for layer in self.layers:
            if end_flag:
                break
            depth += 1
            nxt = []
            for cand in layer_cands:
                if is_root:
                    x, _ = layer(cand.x, data.edge_index)  # updates x, [n, F_dim]
                    new_color, trace, gx = self.get_colors_and_traces(x, batch, trace)
                    nxt.append(Cand(new_color, x, gx, trace))
                    is_root = False
                    gate = None

                else:
                    colors = cand.color
                    #assert sum(len(color) for color in colors) == cand.x.shape[0]
                    ind_batch_colors = [[] for _ in range(self.max_width)]
                    ind_vs = [[] for _ in range(self.max_width)]
                    discrete_count = 0
                    for graph_color in colors:
                        # print(graph_color)
                        # print(self.splitter(graph_color).shape)
                        target_cell = tuple(self.splitter(graph_color))
                        if len(target_cell) == 1:  # discrete
                            for i in range(self.max_width):
                                ind_batch_colors[i].append(graph_color)
                                ind_vs[i].append(-1)
                            discrete_count += 1
                            continue
                        for i, v in enumerate(target_cell):  # target cell, pick first max_width ind vertex
                            if i < self.max_width:
                                ind_graph_color = self.individualize(graph_color, v)
                                ind_batch_colors[i].append(ind_graph_color)
                                ind_vs[i].append(v)

                    if discrete_count == self.bs // self.world_size:  # world_size()
                        nxt = layer_cands
                        end_flag = True
                        break
                    #for batch_color in ind_batch_colors:
                    #    print([len(graph_color) for graph_color in batch_color])
                    ind_batch_colors = [torch.cat(each, dim=0).unsqueeze(-1).to(device) for each in ind_batch_colors if
                                        each != []]

                    for j, batch_color in enumerate(ind_batch_colors):
                        if self.one_hot:
                            batch_color = F.one_hot(batch_color.long(), num_classes=self.max_nodes).int().squeeze(1)
                        ind_x = torch.cat((cand.x, batch_color), dim=1)
                        x, gate = layer(ind_x, data.edge_index)  # updates x, [n, F_dim]
                        new_color, trace, gx = self.get_colors_and_traces(x, batch, v=ind_vs[j], Adjs=Adjs, last_trace=cand.trace)
                        nxt.append(Cand(new_color, x, gx, trace))  # color,trace,gx [bs, ...]

            layer_cands = nxt
            if gate is not None:
                gates.append(gate)
        # if depth > 1:
        # print("depth", depth, "leaves", len(layer_cands))
        # print(layer_cands[0].trace)
        best_idxs = torch.argmax(torch.stack([cand.trace for cand in layer_cands]), dim=0)  # bs
        # print(best_idxs)
        best_x, best_color, best_trace = [], [], []
        for i in range(len(best_idxs)):
            best_x.append(layer_cands[best_idxs[i]].gx[i])
            best_trace.append(layer_cands[best_idxs[i]].trace[i])
            best_color.append(layer_cands[best_idxs[i]].color[i])
        # print(best_trace)
        return torch.cat(best_x, dim=0), torch.stack(best_trace, dim=0), best_color, gates


class GatedGINLayer(nn.Module):
    def __init__(self, in_ch, out_ch, is_root=False):
        super().__init__()
        self.mplayer = GINConv(nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch)))
        self.is_root = is_root
        if not is_root:
            self.alpha = nn.Parameter(torch.tensor(1.0))  # initial gate 1

    def forward(self, x, edge_index):
        if self.is_root:
            gate = 1
        else:
            gate = self.alpha
        y = self.mplayer(x, edge_index)
        return gate * y, gate
