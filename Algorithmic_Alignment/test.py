import os, random, argparse, json, math, pathlib, sys

import networkx as nx
from torch_geometric.utils import to_dense_batch, to_networkx
from torch.optim import AdamW

repo_root = pathlib.Path(__file__).resolve().parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
PATH1 = "./Graphormer"
PATH3 = "./Graphormer/fairseq"
sys.path.append(PATH1)
sys.path.append(PATH3)
#print(sys.path)
from Siamese import Siamese, SiamesePLE
from Dataset import GraphPairDataset, CustomDataLoader, load_samples, create_data, GraphormerDataLoader, ReadingGraphPairs
from Synthesize_dataset import sample_pair_cfi, sample_pair, sample_pair_SR, sample_pair_xor, sample_pair_exp
import sklearn.model_selection
from torch.utils.data import DataLoader, DistributedSampler
import torch, torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = random.randint(1, 100)
random.seed(seed)
torch.manual_seed(seed)
num_data = 100
bs = 16
model_name = "GCN-RNI"
lr = 3e-4
wd = 0
num_layers = 2
workers = 0
g_list, y_list = [], []
nxg_list = []
PLE = False
data_name = "sr"

project_root = pathlib.Path(__file__).resolve().parents[1]  # …/GraphIsomorphism
dataset = ReadingGraphPairs(root=project_root, data_name=data_name)
for i in range(len(dataset)):
    (g1, g2), label = dataset[i]
    g_list.append((g1, g2))
    nxg_list.append((to_networkx(g1),to_networkx(g2)))
    y_list.append(label)
dataset = GraphPairDataset(load_samples(g_list), y_list)

train_ds, test_ds = sklearn.model_selection.train_test_split(
    dataset, test_size=0.2, stratify=y_list, random_state=seed)

dl_kwargs = dict(batch_size=bs,
                 num_workers=workers,
                 pin_memory=True)
a_dl_kwargs = dict(batch_size=1,
                   pin_memory=True)
Loader = CustomDataLoader if model_name != "Graphormer" else GraphormerDataLoader
train_loader = Loader(train_ds, shuffle=True, drop_last=True, **dl_kwargs)
test_loader = Loader(test_ds, shuffle=False, drop_last=True, **dl_kwargs)
a_train_loader = Loader(train_ds, **a_dl_kwargs)
a_val_loader = Loader(test_ds, **a_dl_kwargs)

for (g1, g2), label in zip(nxg_list, y_list):
    res = nx.is_isomorphic(g1,g2)
    if res != label:
        print(res)
        print(label)
print("All done")
device = torch.device(f"cuda:0") if torch.cuda.is_available() else "cpu"
loss_fn = nn.BCEWithLogitsLoss()
if not PLE:
    base = Siamese(model_name, lr=lr, weight_decay=wd, num_layers=num_layers)
    encoder = base.encoder
    encoder = encoder.to(device)
    classifier = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)

    base = base.to(device)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    for i in range(5):
        train_correct = train_total = 0
        encoder.train()
        classifier.train()
        for (g1, g2), labels in train_loader:
            g1, g2 = g1.to(device), g2.to(device)
            #print(g1.in_degree)
            labels = labels.to(device)
            if model_name == "Graphormer":
                h1 = encoder(g1)
                h2 = encoder(g2)

            else:
                h1 = encoder(g1)
                h2 = encoder(g2)
                h1 = global_add_pool(h1, g1.batch)
                h2 = global_add_pool(h2, g2.batch)

            logits = -classifier((h1 - h2).abs()).squeeze(-1)
            preds = (logits > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            #print(preds)
            #print(labels)
            loss = loss_fn(logits, labels)
            # print(loss)
            # print(logits)

            """print("   y   :", labels[:4].tolist())
            print("logits :", logits[:4].cpu().tolist())
            print("prob   :", torch.sigmoid(logits)[:4].cpu().tolist())"""

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_correct = val_total = 0
        val_loss_sum = 0.0
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            for (g1, g2), labels in test_loader:
                g1, g2 = g1.to(device), g2.to(device)
                labels = labels.to(device)
                if model_name == "Graphormer":
                    h1 = encoder(g1)
                    h2 = encoder(g2)

                else:
                    h1 = encoder(g1)
                    h2 = encoder(g2)
                    h1 = global_add_pool(h1, g1.batch)
                    h2 = global_add_pool(h2, g2.batch)
                logits = -classifier((h1 - h2).abs()).squeeze(-1)
                probs = torch.sigmoid(logits)  # (B,) in [0,1]
                preds = (probs > 0.5).float()  # threshold at 0.5
                val_loss = loss_fn(logits, labels)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += val_loss

                """print("   y   :", labels[:4].tolist())
                print("logits :", logits[:4].cpu().tolist())
                print("prob   :", torch.sigmoid(logits)[:4].cpu().tolist())"""

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss_avg = val_loss_sum / val_total
        scheduler.step(val_loss_avg)
        print("acc:", train_acc)
        print("val_acc:", val_acc)

else:
    base = SiamesePLE(model_name, lr=lr, wd=wd, proj_layers=num_layers)
    encoder = base.encoder.to(device)
    base = base.to(device)
    optimizer = AdamW(base.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    for i in range(10):
        train_correct = train_total = 0
        base.train()
        for (g1, g2), labels in train_loader:
            g1, g2 = g1.to(device), g2.to(device)
            labels = labels.to(device)
            loss, preds = base.step(((g1, g2), labels))
            # print(loss)

            """print("   y   :", labels[:4].tolist())
            print("logits :", logits[:4].cpu().tolist())
            print("prob   :", torch.sigmoid(logits)[:4].cpu().tolist())"""

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_correct = val_total = 0
        val_loss_sum = 0.0
        base.eval()
        with torch.no_grad():
            for (g1, g2), labels in test_loader:
                g1, g2 = g1.to(device), g2.to(device)
                labels = labels.to(device)
                val_loss, preds = base.step(((g1, g2), labels))

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += val_loss

                """print("   y   :", labels[:4].tolist())
                print("logits :", logits[:4].cpu().tolist())
                print("prob   :", torch.sigmoid(logits)[:4].cpu().tolist())"""

        print("val_loss", val_loss_sum)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss_avg = val_loss_sum / val_total
        scheduler.step(val_loss_avg)
        print("acc:", train_acc)
        print("val_acc:", val_acc)

from torch_geometric.utils import to_networkx
from networkx.algorithms import weisfeiler_lehman_graph_hash


def wl_hash(G): return weisfeiler_lehman_graph_hash(G)


cnt = 0
for i in range(num_data):
    (G1, G2), y = nxg_list[i], y_list[i]
    cnt += (wl_hash(G1) != wl_hash(G2)) == (y == 0)
print(f"1-WL accuracy on {num_data} pairs:", cnt / num_data)

count, close_count, total = 0, 0, 0
last_h = None
for (g1, g2), labels in a_train_loader:
    g1, g2 = g1.to(device), g2.to(device)
    labels = labels.to(device)
    if model_name == "Graphormer":
        h1 = encoder(g1)
        h2 = encoder(g2)

    else:
        h1 = encoder(g1)
        h2 = encoder(g2)
        h1 = global_add_pool(h1, g1.batch)
        h2 = global_add_pool(h2, g2.batch)
    if last_h is not None:
        print(torch.allclose(h1, last_h, atol=1e-4, rtol=1e-4))
    last_h = h1
    print(h1[:,:10])

    iso = torch.allclose(h1, h2, atol=1e-4, rtol=1e-4)
    if labels.item() == 1 and iso:
        count += 1
    if labels.item() == 0 and not iso:
        count += 1
    if iso:
        close_count += 1
    total += 1

print("train pair-wise correctness", count, "/", total, "=", count / total)
print("train close pairs", close_count, "/", total, "=", close_count / total)

count, close_count, total = 0, 0, 0
if not PLE:
    for (g1, g2), labels in a_val_loader:
        g1, g2 = g1.to(device), g2.to(device)
        labels = labels.to(device)
        if model_name == "Graphormer":
            h1 = encoder(g1)
            h2 = encoder(g2)

        else:
            h1 = encoder(g1)
            h2 = encoder(g2)
            h1 = global_add_pool(h1, g1.batch)
            h2 = global_add_pool(h2, g2.batch)
        iso = torch.allclose(h1, h2, atol=1e-4, rtol=1e-4)
        if labels.item() == 1 and iso:
            count += 1
        if labels.item() == 0 and not iso:
            count += 1
        if iso:
            close_count += 1
        total += 1
else:
    mi, ma = 1, 0
    for (g1, g2), labels in a_val_loader:
        g1, g2 = g1.to(device), g2.to(device)
        labels = labels.to(device)
        z1, z2 = base((g1, g2))
        iso = torch.allclose(z1, z2, atol=1e-4, rtol=1e-4)
        if labels.item() == 1 and iso:
            count += 1
        if labels.item() == 0 and not iso:
            count += 1
        if iso:
            close_count += 1
        total += 1
        iso_min, non_iso_max = base.find_boundary(((g1, g2), labels))
        if mi > iso_min:
            mi = iso_min
        if ma < non_iso_max:
            ma = non_iso_max
    print(f"iso_min_sim:{mi}")
    print(f"non_iso_max_sim:{ma}")
    print(f"current boundary:{(base.criterion.b_theta - base.criterion.b).item()}")
    print(base.criterion.b)
    print(base.criterion.b_theta)

print(f"val pair-wise correctness, PLE: {PLE}", count, "/", total, "=", count / total)
print("val close pairs", close_count, "/", total, "=", close_count / total)
