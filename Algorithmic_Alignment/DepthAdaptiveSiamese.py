import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data
from model import BFS_Refine
import torch.nn.functional as F

class CanonicalNet(nn.Module):
    def __init__(self, d_raw=1, d_hid=128, d_trace=64, k_layers=6, loss_co=1e-3, lr=3e-4, wd=4e-5):
        super().__init__()
        self.loss_co = loss_co
        self.bfs = BFS_Refine(d_raw, d_hid, d_trace, k_layers)
        self.cls = nn.Sequential(nn.Linear(d_trace,64),
                                 nn.ReLU(),
                                 nn.Linear(64,1))
        self.lr = lr
        self.wd = wd

    def forward_one(self, data):
        trace,gates = self.bfs(data)            # trace (64,)   gates (L,)
        return trace, gates

    def forward(self, g1:Data, g2:Data, label=None):
        t1,g1s = self.forward_one(g1)
        t2,g2s = self.forward_one(g2)
        score  = torch.sigmoid(self.cls(torch.abs(t1-t2))).squeeze()
        if label is None:
            return score
        bce = F.binary_cross_entropy(score, label.float())
        depth_pen = (g1s.sum()+g2s.sum()) * self.loss_co
        loss = bce + depth_pen
        return loss, score

    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=self.lr,weight_decay=self.wd,betas=(0.9,0.99))