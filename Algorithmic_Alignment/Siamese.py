import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import sys, os

#from Graphormer.graphormer.models import GraphormerModel
from model import GINModel, SimPLELoss, momentum_update, GCN_RNI, GCN_Pooling, GraphMatchingNetwork
from iter import CanonicalNet_nolearn
from torch_geometric.nn import global_mean_pool, Set2Set, global_add_pool
from argparse import Namespace
import copy


class GraphEncoder(nn.Module):
    def __init__(self, model=None, in_dim=256):
        super().__init__()
        self.model = model
        if model == "Graphormer":
            args = Namespace()
            args.max_nodes = in_dim
            args.num_classes = 2
            args.num_atoms = 100
            args.num_in_degree = 200
            args.num_out_degree = 200
            args.num_edges = 500
            args.num_spatial = 60
            args.num_edge_dis = 60
            args.edge_type = 4
            args.multi_hop_max_dist = 20
            args.pre_layernorm = True
            args.pretrained_model_name = "none"
            #self.encoder = GraphormerModel.build_model(args, task=None)
        elif model == "GIN":
            self.encoder = GINModel(in_dim, 256, 256, 2)
        elif model == "GCN-RNI":
            self.encoder = GCN_RNI(in_dim, hidden_dim=256,
                                   num_layers=2, rni_dim=32)
        elif model == "GCN-Pooling":
            self.encoder = GCN_Pooling(in_dim, hidden_dim=256,
                                       pool_ratio=0.5)
        elif model == "GMN":
            self.encoder = GraphMatchingNetwork(in_dim,hidden_dim=256,out_dim=256,num_layers=2)
        else:
            print("No recognized model!")

    def forward(self, graph):
        if self.model == "GIN":
            r = self.encoder(graph.x, graph.edge_index)
        elif self.model == "Graphormer":
            r = self.encoder(graph)
        elif self.model == "GCN-RNI":
            r = self.encoder(graph.x, graph.edge_index)

        elif self.model == "GCN-Pooling":
            r, batch = self.encoder(graph.x, graph.edge_index, graph.batch)
            return r, batch
        elif self.model == "GMN":
            # graph is a tuple: (g1, g2)
            g1, g2 = graph
            r1, r2, batch1, batch2 = self.encoder(g1, g2)
            return r1, r2, batch1, batch2
        else:
            r = None
        return r


class Siamese(nn.Module):
    def __init__(self,
                 model,
                 num_layers: int = 1,
                 hidden: int = 256,
                 lr: float = 3e-4,
                 weight_decay: float = 4e-5,
                 learn_classifier: bool = False,
                 ):
        super().__init__()

        self.model = model
        self.encoder = GraphEncoder(model)
        self.linear = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layer = num_layers
        self.emb_dim = hidden
        """        for layer in range(self.num_layer):
            self.linear.append(torch.nn.Linear(self.emb_dim*2, self.emb_dim*2))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim*2))"""

        self.learn_classifier = learn_classifier
        if self.learn_classifier:
            self.classifier = nn.Sequential(
                #nn.Linear(self.emb_dim, hidden),
                #nn.ReLU(),
                nn.Linear(hidden, 1)
            )

        #self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.readout = global_add_pool

    # ----- forward pass -------------------------------------------------
    def encode(self, batch):
        return self.encoder(batch)

    def forward(self, pair):
        g1, g2 = pair
        if self.model == "GMN":
            h1, h2, batch1, batch2 = self.encoder((g1, g2))
            h1 = self.readout(h1, batch1)
            h2 = self.readout(h2, batch2)
            #print(h1.shape)
        elif self.model == "GCN-Pooling":
            h1, batch1 = self.encoder(g1)
            h2, batch2 = self.encoder(g2)
            h1 = self.readout(h1, batch1)
            h2 = self.readout(h2, batch2)
            #print(h1.shape)
        else:
            h1 = self.encoder(g1)
            h2 = self.encoder(g2)
            h1 = self.readout(h1, g1.batch)
            h2 = self.readout(h2, g2.batch)

        z = (h1-h2).abs()
        #print(z.sum(dim=-1))
        if self.learn_classifier:
            logits = torch.sigmoid(-self.classifier(z).squeeze(-1))
        else:
            logits = torch.exp(-z.sum(dim=-1).squeeze(-1))

        return logits

    # ----- training / validation ----------------------------------------
    def _shared_step(self, batch):
        (g1, g2), label = batch  # label in {0,1}
        logits = self((g1, g2))
        loss = self.loss_fn(logits, label.float())
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == label.bool()).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self._shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    # ----- optimizer ----------------------------------------------------
    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=self.lr,weight_decay=self.weight_decay,betas=(0.9,0.99))

class SiamesePLE(nn.Module):
    def __init__(
        self,
        model: str,
        emb_dim: int = 256,
        proj_layers: int = 1,
        lr: float = 3e-4,
        wd: float = 4e-5
    ) -> None:
        super().__init__()
        # online / target encoders ------------------------------------------------
        backbone = GraphEncoder(model)
        self.model = model
        self.encoder = backbone

        # 2-layer MLP projection head -------------------------------------------
        mlp: list[nn.Module] = []
        for l in range(proj_layers):
            mlp.append(nn.Linear(2*emb_dim, 2*emb_dim, bias=False))
            mlp.append(nn.BatchNorm1d(2*emb_dim))
            if l < proj_layers - 1:
                mlp.append(nn.ReLU(inplace=True))
        self.project = nn.Sequential(*mlp)
        self.readout = Set2Set(emb_dim, processing_steps=3)

        #self.criterion = SimPLELoss(r=r, alpha=alpha, b=b, b_theta=b_theta)
        self.lr = lr
        self.wd = wd

    # -------------------------------------------------------------------------
    def _encode(self, g, encoder: nn.Module):
        """Encode *one* graph with whichever backbone is supplied."""
        # Node embeddings → projection → graph embedding → ℓ2-norm
        h = encoder(g)                      # [N, emb_dim]
        h = self.readout(h, g.batch)       # [B, emb_dim]
        h = self.project(h)
        return F.normalize(h, dim=-1)      # unit-norm vector

    def _encode_GMN(self, g1, g2, encoder):
        h1, h2 = encoder((g1, g2))
        h1, h2 = self.readout(h1, g1.batch), self.readout(h2, g2.batch)
        h1, h2 = self.project(h1), self.project(h2)
        return F.normalize(h1, dim=-1), F.normalize(h2, dim=-1)

    def _encode_GCN_Pooling(self, g, encoder: nn.Module):
        """Encode *one* graph with whichever backbone is supplied."""
        # Node embeddings → projection → graph embedding → ℓ2-norm
        h, batch = encoder(g)                      # [N, emb_dim]
        h = self.readout(h, batch)       # [B, emb_dim]
        h = self.project(h)
        return F.normalize(h, dim=-1)      # unit-norm vector

    def forward(self, batch):
        """Return (z_q1, z_q2, z_k1, z_k2)"""
        (g1, g2), same_label = batch
        if self.model == "GCN-Pooling":
            z1 = self._encode_GCN_Pooling(g1, self.encoder)
            z2 = self._encode_GCN_Pooling(g2, self.encoder)

        elif self.model == "GMN":
            z1, z2 = self._encode_GMN(g1, g2, self.encoder)

        else:
            z1 = self._encode(g1, self.encoder)
            z2 = self._encode(g2, self.encoder)

        return z1, z2

    def find_boundary(self, batch):
        with torch.no_grad():
            (g1, g2), same_label = batch  # `same_label` *(B,)∈{0,1}*
            z1, z2 = self((g1, g2))
            inner = (z1*z2).sum(dim=-1)
            s = inner
            if same_label.shape[0] > 1:
                iso_minimum = s[same_label==1].min()
                non_iso_maximum = s[same_label==0].max()
            else:
                if same_label[0] == 1:
                    return s[0], 0
                else:
                    return 1, s[0]
        return iso_minimum, non_iso_maximum

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, betas=(0.9,0.99))


