import os, csv, time, pathlib, multiprocessing as mp
import networkx as nx
from queue import Empty
from contextlib import closing
import math

# ---------------------------------------------------------------------------
DATA_DIR    = pathlib.Path("../benchmark_new")      # directory with 0_1, 0_2 ...
OUT_LOG     = pathlib.Path("./benchmark_new_noPruning.csv")
TIMEOUT_S   = 10800       # seconds for canonical-form run
GT_LIMIT    = 10800          # optional ground-truth timeout
MAX_WORKERS = min(128-1, mp.cpu_count()-1)   # hard cap

# ---------- lightweight DIMACS reader ---------------------------------------
def read_dimacs(path):
    G = nx.Graph()
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("p"):
                n = int(ln.split()[2]);  G.add_nodes_from(range(n))
            elif ln.startswith("e"):
                u, v = (int(x)-1 for x in ln.split()[1:3])
                G.add_edge(u, v)
    return G

# ---------- worker ----------------------------------------------------------
def canon_worker(pair_idx, dir_path, out_q):
    """
    Executed in a separate process.
    Sends the result dict through out_q.
    """
    import main   # heavy imports kept local to the worker
    d = pathlib.Path(dir_path)
    f1, f2 = d/f"{pair_idx}_1", d/f"{pair_idx}_2"
    try:
        G1, G2 = read_dimacs(f1), read_dimacs(f2)
        t0 = time.perf_counter()
        iso = main.is_isomorphic(G1, G2)
        runtime = time.perf_counter() - t0
    except Exception as exc:
        iso, runtime = f"error:{exc}", -1

    # optional short truth run
    truth, truth_to = "noRun", "noRun"
    try:
        truth = nx.is_isomorphic(G1, G2)
        truth_to = False
    except Exception:
        truth, truth_to = "unknown", True
    finally:

        print({
            "pair_index":  pair_idx,
            "isomorphic":  iso,
            "ground_truth": truth,
            "canon_timeout": False,
            "truth_timeout": truth_to,
            "runtime_s": f"{runtime:.4f}" if runtime >= 0 else "err"
        })
        out_q.put({
            "pair_index":  pair_idx,
            "isomorphic":  iso,
            "ground_truth": truth,
            "canon_timeout": False,
            "truth_timeout": truth_to,
            "runtime_s": f"{runtime:.4f}" if runtime >= 0 else "err"
        })

# ---------- driver ----------------------------------------------------------
def benchmark(n_pairs=451, data_dir=DATA_DIR, csv_path=OUT_LOG):
    sem   = mp.Semaphore(MAX_WORKERS)      # concurrency limiter
    tasks = []                             # (proc, queue, start_time)

    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(
            fh, ["pair_index","isomorphic","ground_truth",
                 "canon_timeout","truth_timeout","runtime_s"])
        wr.writeheader(); fh.flush()

        for i in range(n_pairs):
            if not (data_dir/f"{i}_1").exists():
                continue                   # silently skip missing pair

            """G1 = read_dimacs(f1)
            if G1.number_of_nodes() > 1000:
                print(f"{pair_index}:skipped")
                continue"""

            # ----- launch a worker --------------------------------------
            sem.acquire()                  # block if 128 already running
            q = mp.Queue()
            p = mp.Process(target=canon_worker,
                           args=(i, str(data_dir), q))
            p.start()
            tasks.append((i, p, q, time.perf_counter()))

            # ----- collect finished / timed-out workers -----------------
            tasks = _harvest(tasks, wr, fh, sem)

        # wait for all remaining workers
        while tasks:
            tasks = _harvest(tasks, wr, fh, sem, final=True)

    print("Benchmark finished ➜", csv_path)

# ---------- helper to reap children ----------------------------------------
def _harvest(task_list, writer, fh, sem, final=False):
    """
    Try to join() each child for up to 0 s (poll) unless `final` is True,
    in which case we wait whichever is shorter: remaining timeout or 1 s.
    Returns the pruned task list.
    """
    now  = time.perf_counter()
    keep = []

    for idx, proc, q, t0 in task_list:
        limit = TIMEOUT_S - (now - t0)
        join_t = max(0.0, min(1.0, limit)) if not final else max(0.0, min(1.0, limit))
        proc.join(join_t)

        if proc.is_alive() and limit <= 0:             # ------ timeout
            proc.kill(); proc.join()
            writer.writerow({
                "pair_index": idx, "isomorphic": "TIMEOUT",
                "ground_truth": "unknown",
                "canon_timeout": True, "truth_timeout": True,
                "runtime_s": f">{TIMEOUT_S}"
            }); fh.flush()
            sem.release()                              # free a slot
        elif not proc.is_alive():                      # ------ finished
            try:
                res = q.get_nowait()  # non-blocking
            except Empty:
                res = {"pair_index": idx,
                       "isomorphic": "worker_crashed",
                       "ground_truth": "unknown",
                       "canon_timeout": False,
                       "truth_timeout": False,
                       "runtime_s": "err"
                    }
            writer.writerow(res)
            fh.flush()
            msg = res['runtime_s']
            print(f"[{idx:03}] iso={res['isomorphic']} truth={res['ground_truth']}  {msg}")
            sem.release()
        else:                                          # still running
            keep.append((idx, proc, q, t0))

    return keep

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    benchmark()
