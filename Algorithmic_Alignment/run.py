import gc
import itertools
import subprocess
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import sklearn.model_selection
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj
import os, random, argparse, json, math, pathlib, sys

repo_root = pathlib.Path(__file__).resolve().parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
PATH1 = "../Graphormer"
PATH3 = "../Graphormer/fairseq"
sys.path.append(PATH1)
sys.path.append(PATH3)
import torch, torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from utils import EarlyStopper, ddp_setup, cleanup, pad_sparse_matrix, average_gradients
from Dataset import GraphPairDataset, CustomDataLoader, load_samples, create_data, ReadingGraphPairs
from Siamese import Siamese, SiamesePLE
from tqdm import tqdm
import socket

def run_epoch(model, loader, rank, optimizer=None, PLE=False):
    loss_fn = nn.BCELoss()
    train = optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    world_size = dist.get_world_size()
    running_loss, all_logits, all_labels, preds = 0.0, [], [], []
    if not PLE:
        for (g1, g2), labels in loader:
            g1, g2, labels = (x.to(rank) for x in (g1, g2, labels))
            if train:
                model.eval()
                logits = model((g1, g2))
                model.train()
            else:
                logits = model((g1, g2))
            loss = loss_fn(logits, labels)
            if train:
                model.zero_grad()
                loss.backward()
                # average_gradients(model)
                optimizer.step()

            batch_loss = loss.detach()
            dist.all_reduce(batch_loss, op=dist.ReduceOp.SUM)
            running_loss += batch_loss / world_size
            all_logits.append(logits.detach())
            all_labels.append(labels)
        logits = torch.cat(all_logits)
        preds = (logits > 0.5)
    else:
        for (g1, g2), labels in loader:
            g1, g2, labels = (x.to(rank) for x in (g1, g2, labels))
            loss, pred = model(((g1, g2), labels))

            if train:
                model.zero_grad()
                loss.backward()
                # average_gradients(model)
                optimizer.step()
                with torch.no_grad():
                    _, pred = model((g1, g2))

            batch_loss = loss.detach()
            dist.all_reduce(batch_loss, op=dist.ReduceOp.SUM)
            running_loss += batch_loss / world_size
            preds.append(pred.detach())
            all_labels.append(labels)
        preds = torch.cat(preds)

    labels = torch.cat(all_labels)
    #if rank == 0:
    #    print("pred", preds.sum(), preds[:4])
    #    print("label",labels.sum(), labels[:4])
    preds_list = [torch.empty_like(preds) for _ in range(world_size)]
    labels_list = [torch.empty_like(labels) for _ in range(world_size)]
    dist.all_gather(preds_list, preds.cuda(rank))
    dist.all_gather(labels_list, labels.cuda(rank))

    if rank == 0:  # metrics only once
        preds_all = torch.cat(preds_list).cpu()
        labels_all = torch.cat(labels_list).cpu()
        #print("pred", preds_all.sum(), preds_all[:20])
        #print("label", labels_all.sum(), labels_all[:20])
        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all)
        mean_loss = running_loss / len(loader)
    else:
        acc = f1 = mean_loss = math.nan  # ignored by caller

    metrics = torch.tensor([mean_loss, acc, f1], device='cuda')
    dist.broadcast(metrics, src=0)
    mean_loss, acc, f1 = metrics.tolist()
    return mean_loss, acc, f1


def run_single_config(train_ds, test_ds, lr, wd, jid=0, model_name="GIN", data_name="syn", epochs=15, bs=32,
                      num_layers=2, workers=4, seed=42, PLE=False, rank=0, world_size=4):
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    test_sampler = DistributedSampler(test_ds, shuffle=False)

    dl_kwargs = dict(batch_size=bs // world_size,
                     num_workers=workers,
                     persistent_workers=False,
                     pin_memory=True)

    train_loader = CustomDataLoader(train_ds, sampler=train_sampler, drop_last=True, **dl_kwargs)
    test_loader = CustomDataLoader(test_ds, sampler=test_sampler, **dl_kwargs)

    if rank == 0:
        os.makedirs("runs", exist_ok=True)
        with open(f"runs/log_{model_name}_{data_name}_{PLE}_{jid}.txt", "w") as f:
            f.write(f"lr:{lr}, wd:{wd}, layers:{num_layers}, bs:{bs}\n")

    seed = seed
    random.seed(seed)
    torch.manual_seed(seed)

    # ddp_setup(rank, world_size)

    # -------- model --------
    if not PLE:
        base = Siamese(model_name, lr=lr, weight_decay=wd, num_layers=num_layers, learn_classifier=True)
    else:
        base = SiamesePLE(model_name, lr=lr, wd=wd, proj_layers=num_layers)

    base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
    model = base.to(rank)

    # if rank == 0:
    #    print("start ddp")
    model = DDP(model, device_ids=[rank])
    if rank == 0:
        print("model parallelized")
    optimizer = base.configure_optimizers()  # DDP wraps only fwd/bwd
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=2)

    history = defaultdict(list)
    true_epoch = None
    patience, bad_epochs, wait = 7, 3, 0
    stopper = EarlyStopper(patience=patience, bad_epochs=bad_epochs, mode='min')
    last_train_loss = None
    for epoch in tqdm(range(1, epochs + 1), desc=f"cfg={lr, wd}"):
        # print(f"rank {rank} reach {epoch}")
        train_sampler.set_epoch(epoch)

        te_loss, te_acc, te_f1 = run_epoch(model, test_loader, rank, PLE=PLE)
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, rank, optimizer, PLE=PLE)
        #te_loss, te_acc, te_f1 = run_epoch(model, test_loader, rank, PLE=PLE)

        scheduler.step(tr_loss)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["train_f1"].append(tr_f1)
        history["test_f1"].append(te_f1)
        if rank == 0:
            with open(f"runs/log_{model_name}_{data_name}_{PLE}_{jid}.txt", "a") as f:
                log = f"[{epoch:02d}/{epochs}] | train loss {tr_loss:.3f} | acc {tr_acc:.3f} | f1 {tr_f1:.3f} | val loss {te_loss:.3f} | acc {te_acc:.3f} | f1 {te_f1:.3f}\n"
                f.write(log)

        if stopper.step(tr_loss, te_loss):
            print(f"Early stop at epoch {epoch}")
            # if stopper.wait_val > patience:
            # print("Val not improved")
            # else:
            # print("Train loss increased")
            break

    # print(f"rank {rank} reach barrier")
    dist.barrier()
    del model, optimizer, scheduler, train_loader, train_sampler, test_loader, test_sampler
    torch.cuda.empty_cache()
    gc.collect()
    # cleanup()

    idx = np.argmax(history["test_acc"])
    return history["train_acc"][idx], history["test_acc"][idx]


def launch(data, model, PLE):
    GRID = list(itertools.product(
        [1e-4, 3e-4, 1e-3],  # lrs
        [0, 4e-5, 1e-4],  # wds
    ))

    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Start running on rank {rank}.")
    seed = 42
    g_list, y_list = [], []
    project_root = pathlib.Path(__file__).resolve().parents[1]  # …/GraphIsomorphism
    dataset = ReadingGraphPairs(root=project_root, data_name=data)
    for i in range(len(dataset)):
        (g1, g2), label = dataset[i]
        g_list.append((g1, g2))
        y_list.append(label)
    dataset = GraphPairDataset(load_samples(g_list), y_list)
    if rank == 0:
        print(f"Dataset of length {len(dataset)} loaded.")
    train_ds, test_ds = sklearn.model_selection.train_test_split(
        dataset, test_size=0.25, shuffle=True, stratify=y_list, random_state=seed)

    if rank == 0:
        pos_count = 0
        neg_count = 0
        for (g1,g2), label in train_ds:
            if label.item() == 0:
                neg_count += 1
            else:
                pos_count += 1
        print(f"Counts of neg/pos labels in test dataset:{neg_count} {pos_count} (Should be around 192)")

    tr_acc_list, te_acc_list = [], []
    for job_id, (lr, wd) in enumerate(GRID):
        if rank == 0:
            print(f"job:{job_id}, lr:{lr},wd:{wd} starts")
        tr_acc, te_acc = run_single_config(train_ds=train_ds, test_ds=test_ds,
                                           lr=lr, wd=wd, jid=job_id, data_name=data,
                                           model_name=model, PLE=PLE, rank=rank)
        if rank == 0:
            tr_acc_list.append(tr_acc)
            te_acc_list.append(te_acc)

    if rank == 0:
        os.makedirs("res", exist_ok=True)
        best_te_acc, cfg_index = max(te_acc_list), np.argmax(te_acc_list)
        with open(f"res/{model}_{data}_{PLE}.txt", "w") as f:
            log = f"best train/val acc: {tr_acc_list[cfg_index]}, {best_te_acc} " \
                  f"with lr:{GRID[cfg_index][0]}, wd:{GRID[cfg_index][1]} "
            f.write(log)
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    log_dir = pathlib.Path("runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="GIN")
    p.add_argument("--PLE", type=str, default="False")
    p.add_argument("--data", type=str, default="syn")
    args = p.parse_args()

    if args.PLE == "False":
        PLE = False
    elif args.PLE == "True":
        PLE = True
    else:
        PLE = False
    print(f"Training on config:{args.model},{PLE},{args.data} starts")
    launch(args.data, args.model, PLE)
