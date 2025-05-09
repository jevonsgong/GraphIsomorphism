import os, csv, time, multiprocessing as mp
import networkx as nx
import math

DATA_DIR   = "./benchmark_new"
OUTPUT_LOG = "benchmark_new.csv"
TIMEOUT_S  = math.inf         # canonical-form limit
GT_LIMIT   = math.inf           # ground-truth limit

# ---------- helpers --------------------------------------------------

def read_dimacs(path):
    G = nx.Graph()
    with open(path) as f:
        for ln in f:
            if ln.startswith("p"):
                n = int(ln.split()[2]); G.add_nodes_from(range(n))
            elif ln.startswith("e"):
                u,v = map(lambda x:int(x)-1, ln.split()[1:])
                G.add_edge(u,v)
    return G

def canon_worker(path1, path2, q):
    from main import is_isomorphic       # imports inside child
    q.put(is_isomorphic(read_dimacs(path1), read_dimacs(path2)))

def truth_worker(path1, path2, q):
    q.put(nx.is_isomorphic(read_dimacs(path1), read_dimacs(path2)))

# ---------- benchmark ------------------------------------------------

def benchmark(pairs=451, data_dir=DATA_DIR, log_path=OUTPUT_LOG):
    with open(log_path, "w", newline="") as csvf:
        wr = csv.DictWriter(csvf,
              ["pair","isomorphic","ground_truth",
               "canon_timeout","truth_timeout","runtime"])
        wr.writeheader()

        for i in range(pairs):
            f1,f2 = [os.path.join(data_dir,f"{i}_{t}") for t in (1,2)]
            if not (os.path.exists(f1) and os.path.exists(f2)):
                continue
            #G1 = read_dimacs(f1)
            """if G1.number_of_nodes() > 500:
                print(f"{i}:skipped")
                wr.writerow(dict(pair=i, isomorphic="skipped", ground_truth="skipped",
                                 canon_timeout="skipped", truth_timeout="skipped",
                                 runtime="skipped"))
                continue"""
            # --- canonical-form process ---
            q = mp.Queue()
            p = mp.Process(target=canon_worker, args=(f1,f2,q))
            t0 = time.perf_counter();   p.start()
            p.join(TIMEOUT_S)
            canon_to = p.is_alive()
            if canon_to:
                p.kill(); p.join()
                iso = "TIMEOUT"
            else:
                iso = q.get()
            canon_time = time.perf_counter() - t0

            # --- ground truth (short limit) ---
            gt_q = mp.Queue()
            gt_p = mp.Process(target=truth_worker, args=(f1,f2,gt_q))
            gt_p.start();   gt_p.join(GT_LIMIT)
            truth_to = gt_p.is_alive()
            if truth_to:
                gt_p.kill(); gt_p.join()
                truth = "unknown"
            else:
                truth = gt_q.get()

            wr.writerow(dict(pair=i, isomorphic=iso, ground_truth=truth,
                             canon_timeout=canon_to, truth_timeout=truth_to,
                             runtime=f"{canon_time:.3f}"))
            status = "TIMEOUT" if canon_to else f"{canon_time:.2f}s"
            print(f"[{i:03d}] iso={iso} truth={truth}  {status}")

    print("Benchmark finished ->", log_path)

# --------------------------------------------------------------------
if __name__ == "__main__":
    benchmark()
