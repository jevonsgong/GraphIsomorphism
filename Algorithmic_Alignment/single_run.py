import json, argparse, torch
import os, random, argparse, json, math, pathlib, sys
repo_root = pathlib.Path(__file__).resolve().parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from Algorithmic_Alignment.run import run_single_config

p = argparse.ArgumentParser()
p.add_argument("--cfg", type=str, required=True)
p.add_argument("--jid", type=int, default=0)
p.add_argument("--model", type=str, default="GIN")
p.add_argument("--PLE", type=str, default="False")
p.add_argument("--data", type=str, default="syn")
p.add_argument("--rank", type=int, default=0)
args = p.parse_args()

cfg = json.loads(args.cfg)

if args.PLE == "False":
    PLE = False
elif args.PLE == "True":
    PLE = True
else:
    PLE = False
try:
    val_loss = run_single_config(lr=cfg["lr"], wd=cfg["wd"], jid=args.jid, data_name=args.data, model_name=args.model, PLE=PLE, rank=args.rank)
finally:
    torch.cuda.empty_cache()
print(f"FINISHED {cfg}  val_loss={val_loss}")