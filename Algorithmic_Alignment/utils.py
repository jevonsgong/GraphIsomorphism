import os, torch
import torch.distributed as dist
import math
import socket

class EarlyStopper:
    """
    mode='min' (loss) or 'max' (accuracy); stops if
    • val metric doesn't improve for `patience` epochs, OR
    • train metric increases for `bad_epochs` consecutive epochs.
    """
    def __init__(self, patience=3, bad_epochs=2, mode='min', min_delta=1e-3):
        self.patience    = patience
        self.bad_epochs  = bad_epochs
        self.mode        = mode
        self.min_delta   = min_delta
        self.best_val    = math.inf if mode == 'min' else -math.inf
        self.wait_val    = 0
        self.prev_train  = None
        self.bad_counter = 0

    def improved(self, current, best):
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def step(self, train_metric, val_metric):
        # ---- validation part -----------------------------------------------
        if self.improved(val_metric, self.best_val):
            self.best_val, self.wait_val = val_metric, 0
        else:
            self.wait_val += 1

        # ---- training-divergence part --------------------------------------
        if self.prev_train is not None and \
           not self.improved(train_metric, self.prev_train):
            self.bad_counter += 1
        else:
            self.bad_counter = 0
        self.prev_train = train_metric

        # ---- stop? ----------------------------------------------------------
        return (self.wait_val >= self.patience) or (self.bad_counter >= self.bad_epochs)

def cleanup():
    dist.destroy_process_group()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= size


def pad_sparse_matrix(sparse_matrix, target_size):
    """
    Pads a sparse matrix to the target size by adding zero rows/columns.
    Args:
        sparse_matrix (torch.sparse_coo_tensor): Sparse matrix to pad.
        target_size (tuple): Desired size (rows, cols).
    """
    current_size = sparse_matrix.size()

    # If the sparse matrix is already the correct size, return it as is
    if current_size == target_size:
        return sparse_matrix

    indices = sparse_matrix.coalesce().indices()
    values = sparse_matrix.coalesce().values()

    # Adjust indices for padding
    pad_indices = indices[:, indices[0] < target_size[0]]
    pad_indices = pad_indices[:, indices[1] < target_size[1]]

    # Create a new sparse tensor with the padded size
    padded_matrix = torch.sparse_coo_tensor(pad_indices, values, size=target_size)
    return padded_matrix
