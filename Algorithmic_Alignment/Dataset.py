from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch_geometric.utils import from_networkx, to_dense_batch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.data import Data
from typing import Tuple, List
from pathlib import Path
import re
import os


class GraphPairDataset(Dataset):
    def __init__(self, graph_pairs, labels):
        """
        graph_pairs: List of tuples, where each tuple contains
                     (graph_1_data, graph_2_data, graph_1_list_of_matrix, graph_2_list_of_matrix),
                     and all elements are NumPy arrays.
        labels: List of binary labels corresponding to each graph pair.
        """
        self.graph_pairs = graph_pairs
        self.labels = labels

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        graph_1_data, graph_2_data = self.graph_pairs[idx]
        label = self.labels[idx]

        #graph_1_nodes_tensor = torch.tensor(graph_1_node_idx).float()
        #graph_2_nodes_tensor = torch.tensor(graph_2_node_idx).float()

        #graph_1_matrix_tensor = [sparse_to_torch(matrix) for matrix in graph_1_matrix]
        #graph_2_matrix_tensor = [sparse_to_torch(matrix) for matrix in graph_2_matrix]

        # print(graph_1_data.x)
        return (graph_1_data, graph_2_data), torch.tensor(label, dtype=torch.float)


class ReadingGraphPairs(Dataset):
    """
    Works with files saved as
        <root>/<data_name>/<data_name>_graphs_{pair_id}_{side}_{n}.pt
    where   side = 1  (first graph)  or 2  (second graph).

    Example:  cfi/cfi_graphs_17_1_152_0.pt
              cfi/cfi_graphs_17_2_152_0.pt
    """

    _rx = re.compile(
        r"^(?P<name>.+)_graphs_"  # data_name_
        r"(?P<pair>\d+)_"  # pair id
        r"(?P<side>[12])_"  # 1 or 2
        r"(?P<n>\d+)_"  # optional node count
        r"(?P<label>[01])\.pt$"  # class label
    )

    def __init__(self, root: str | Path, data_name: str, transform=None):
        self.root      = Path(root) / data_name
        self.transform = transform

        # ---------- gather file paths --------------------------------------
        pairs: dict[int, dict[int, Path]] = {}
        labels: dict[int, int] = {}
        for p in self.root.glob(f"{data_name}_graphs_*_*.pt"):
            m = self._rx.match(p.name)
            if not m:
                continue
            pid  = int(m["pair"])
            side = int(m["side"])          # 1 or 2
            lbl = int(m["label"])
            pairs.setdefault(pid, {})[side] = p
            labels[pid] = lbl

        # keep only complete pairs (both side-1 and side-2 present)
        self._pair_paths: List[Tuple[Path, Path, int]] = []
        for pid, d in sorted(pairs.items()):
            if 1 in d and 2 in d:
                self._pair_paths.append((d[1], d[2], labels[pid]))

    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._pair_paths)

    # ----------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[Tuple[Data, Data], int]:
        path1, path2, label = self._pair_paths[idx]

        g1: Data = torch.load(path1)
        g2: Data = torch.load(path2)

        if self.transform is not None:
            g1 = self.transform(g1)
            g2 = self.transform(g2)

        return (g1, g2), label


def sparse_to_torch(sparse_matrix):
    # Convert scipy sparse matrix to COO format (required by PyTorch)
    sparse_matrix = sparse_matrix.tocoo()

    # Create PyTorch sparse tensor
    # indices = torch.LongTensor([sparse_matrix.row, sparse_matrix.col])
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))

    # Then convert to a LongTensor
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(sparse_matrix.data)
    shape = sparse_matrix.shape

    sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
    return sparse_tensor

def create_data(G):
    node_features = []
    for node in G.nodes:
        in_neighbors = G.degree(node)
        # out_neighbors = G.out_degree(node)
        # total_neighbors = len(list(G.neighbors(node)))
        # print(G.neighbors(node))

        node_features.append(in_neighbors)
        # print(node_features)
    node_features = torch.tensor(node_features, dtype=torch.long)
    data = from_networkx(G)
    num_features = 256
    data.x = F.one_hot(node_features, num_classes=num_features).float()
    #print(data.x.shape)
    #data.y = label

    # data.y = 1
    return data

def pad_square(t, target_n, pad_value):
    # t: (C, n, n) or (n, n)
    pad = target_n - t.size(-1)
    if pad == 0:
        return t
    if t.dim() == 3:                           # (C, n, n)
        return F.pad(t, (0, pad, 0, pad), value=pad_value)
    else:                                      # (n, n)
        return F.pad(t, (0, pad, 0, pad), value=pad_value)

def load_samples(dataset: list):
    samples_list = []
    for graph_data in dataset:
        num_nodes_1 = graph_data[0].x.shape[0]
        num_nodes_2 = graph_data[1].x.shape[0]
        edge_index_1 = graph_data[0].edge_index
        edge_index_2 = graph_data[1].edge_index

        adjacency_matrix_1 = to_dense_adj(edge_index_1, max_num_nodes=num_nodes_1)[0]  # Shape (num_nodes, num_nodes)
        adjacency_matrix_2 = to_dense_adj(edge_index_2, max_num_nodes=num_nodes_2)[0]  # Shape (num_nodes, num_nodes)
        spatial_pos_1 = torch.where(adjacency_matrix_1 > 0, adjacency_matrix_1,
                                    torch.tensor(float('inf')))  # Replace 0s with infinity
        spatial_pos_1.fill_diagonal_(0)  # Self-loops have 0 distance
        spatial_pos_2 = torch.where(adjacency_matrix_2 > 0, adjacency_matrix_2,
                                    torch.tensor(float('inf')))  # Replace 0s with infinity
        spatial_pos_2.fill_diagonal_(0)  # Self-loops have 0 distance
        graph_data[0].__setattr__('spatial_pos', spatial_pos_1)
        graph_data[1].__setattr__('spatial_pos', spatial_pos_2)

        num_edges_1 = edge_index_1.size(1)
        num_edges_2 = edge_index_2.size(1)
        edge_dim = 4  # Number of features per edge
        graph_data[0].__setattr__('edge_input', torch.randn((num_edges_1, edge_dim)))
        graph_data[1].__setattr__('edge_input', torch.randn((num_edges_2, edge_dim)))

        num_edge_types = 4
        graph_data[0].__setattr__('edge_type', torch.randint(0, num_edge_types, (num_edges_1,)))
        graph_data[1].__setattr__('edge_type', torch.randint(0, num_edge_types, (num_edges_2,)))

        graph_data[0].__setattr__('attn_edge_type', torch.zeros((1, num_nodes_1, num_nodes_1)))
        graph_data[1].__setattr__('attn_edge_type', torch.zeros((1, num_nodes_2, num_nodes_2)))

        # print(num_nodes)
        # sys.exit()
        graph_data[0].__setattr__('attn_bias', torch.zeros((1, num_nodes_1 + 1, num_nodes_1 + 1)))
        graph_data[1].__setattr__('attn_bias', torch.zeros((1, num_nodes_2 + 1, num_nodes_2 + 1)))

        graph_data[0].__setattr__('in_degree', graph_data[0].x.clone())
        graph_data[1].__setattr__('out_degree', graph_data[0].x.clone())

        graph_data[0].__setattr__('in_degree', graph_data[1].x.clone())
        graph_data[1].__setattr__('out_degree', graph_data[1].x.clone())
        # print(graph_data[0].in_degrees)

        sample = (
            graph_data[0], graph_data[1],
        )
        samples_list.append(sample)

    return samples_list

def inspect_batch(data_list):
    shapes = defaultdict(set)
    for idx, d in enumerate(data_list):
        for key, val in d:
            if torch.is_tensor(val):
                shapes[key].add(tuple(val.shape))
    print(">>> shapes seen in this mini-batch:")
    for k, v in shapes.items():
        if len(v) > 1:
            print(f" • {k}: {sorted(v)}")   # varies → potential culprit


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        graph_1_data_list = []
        graph_2_data_list = []
        labels = []
        for (g1, g2), label in batch:
            graph_1_data_list.append(g1)
            graph_2_data_list.append(g2)
            if isinstance(label, float):
                labels.append(torch.tensor(label))
            else:
                labels.append(label)
        #inspect_batch(graph_1_data_list)
        graph_1_data_batch = Batch.from_data_list(graph_1_data_list,exclude_keys=['attn_bias', 'attn_edge_type', 'spatial_pos'])
        graph_2_data_batch = Batch.from_data_list(graph_2_data_list,exclude_keys=['attn_bias', 'attn_edge_type', 'spatial_pos'])

        attn_bias = []
        edge_type = []
        spatial_pos = []
        PAD_VAL_BIAS = 0
        PAD_VAL_EDGETY = 0
        PAD_VAL_SPOS = torch.inf
        for data_list, batch in zip([graph_1_data_list,graph_2_data_list],[graph_1_data_batch,graph_2_data_batch]):
            max_n = max(d.num_nodes for d in data_list)
            for d in data_list:
                attn_bias.append(pad_square(d.attn_bias, max_n, PAD_VAL_BIAS))
                edge_type.append(pad_square(d.attn_edge_type, max_n, PAD_VAL_EDGETY))
                spatial_pos.append(pad_square(d.spatial_pos, max_n, PAD_VAL_SPOS))

            batch.attn_bias = torch.stack(attn_bias)  # (B, 1, N, N)
            batch.attn_edge_type = torch.stack(edge_type)  # (B, 1, N, N)
            batch.spatial_pos = torch.stack(spatial_pos)  # (B, N, N)
            batch.max_n = max_n
        labels_batch = torch.stack(labels)
        return (graph_1_data_batch,graph_2_data_batch), labels_batch

class GraphormerDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(self, batch):
        graph_1_data_list = []
        graph_2_data_list = []
        labels = []
        for (g1, g2), label in batch:
            graph_1_data_list.append(g1)
            graph_2_data_list.append(g2)
            labels.append(label)
        #inspect_batch(graph_1_data_list)
        graph_1_data_batch = Batch.from_data_list(graph_1_data_list,exclude_keys=['attn_bias', 'attn_edge_type', 'spatial_pos'])
        graph_2_data_batch = Batch.from_data_list(graph_2_data_list,exclude_keys=['attn_bias', 'attn_edge_type', 'spatial_pos'])
        labels_batch = torch.stack(labels)
        attn_bias = []
        edge_type = []
        spatial_pos = []
        PAD_VAL_BIAS = 0
        PAD_VAL_EDGETY = 0
        PAD_VAL_SPOS = torch.inf
        for data_list, batch in zip([graph_1_data_list,graph_2_data_list],[graph_1_data_batch,graph_2_data_batch]):
            max_n = max(d.num_nodes for d in data_list)
            for d in data_list:
                attn_bias.append(pad_square(d.attn_bias, max_n, PAD_VAL_BIAS))
                edge_type.append(pad_square(d.attn_edge_type, max_n, PAD_VAL_EDGETY))
                spatial_pos.append(pad_square(d.spatial_pos, max_n, PAD_VAL_SPOS))

            batch.attn_bias = torch.stack(attn_bias)  # (B, 1, N, N)
            batch.attn_edge_type = torch.stack(edge_type)  # (B, 1, N, N)
            batch.spatial_pos = torch.stack(spatial_pos)  # (B, N, N)
            batch.max_n = max_n
        graph_1_data_batch.x = to_dense_batch(graph_1_data_batch.x,graph_1_data_batch.batch)[0]
        graph_2_data_batch.x = to_dense_batch(graph_2_data_batch.x,graph_2_data_batch.batch)[0]
        return (graph_1_data_batch, graph_2_data_batch), labels_batch

if __name__ == "__main__":
    from Synthesize_dataset import *

    i = 0
    graph_data_list, labels_list = [], []
    while i < 10:
        iso_target = i < 5
        G1, G2, is_iso = sample_pair(iso_prob=1.0 if iso_target else 0.0)
        G1, G2 = create_data(G1), create_data(G2)
        graph_data_list.append((G1,G2))
        labels_list.append(is_iso)
        i += 1
    graph_pairs = load_samples(graph_data_list)
    print(graph_pairs[0])
    dataset = GraphPairDataset(graph_pairs,labels_list)
    dataloader = CustomDataLoader(dataset)
