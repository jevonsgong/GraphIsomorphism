import concurrent.futures as cf
import csv
import pathlib
import re
import time
import resource

import networkx as nx

from main import is_isomorphic

DATA_DIR      = pathlib.Path("../benchmark_new")
OUT_LOG       = pathlib.Path("./benchmark_new_withPruning.csv")
TIMEOUT       = 18000            # seconds per pair
MAX_NODES     = 1000
MAX_WORKERS   = 16
MEM_LIMIT  = 16 * 1024**3


def impose_limits():
    """Called in the worker just after fork/spawn."""
    # wall-clock timeout handled by alarm in worker function
    # RLIMIT_AS = virtual memory incl. swap
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    # RLIMIT_NOFILE just in case
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))

def read_dimacs(path):
    """Minimal DIMACS-like reader (p/e lines only, 1-based)."""
    G = nx.Graph()
    with path.open("r") as f:
        for line in f:
            if not line or line[0] == "c":
                continue
            if line[0] == "p":
                _, _, nv, _ = line.split()
                G.add_nodes_from(range(int(nv)))
            elif line[0] == "e":
                _, u, v = line.split()
                G.add_edge(int(u) - 1, int(v) - 1)
    return G

def run_pair(pair_index, f1, f2):
    """Worker for one graph pair."""
    try:
        G1 = read_dimacs(f1)
        G2 = read_dimacs(f2)
    except Exception as exc:
        return dict(pair_index=pair_index,
                    isomorphic=f"read_error:{exc}")

    '''if G1.number_of_nodes() > MAX_NODES:
        print(f"[{pair_index:04}] skipped (too large)")
        return dict(pair_index=pair_index,
                    file1=f1.name,
                    file2=f2.name,
                    skipped=True)'''
    if G1.number_of_nodes()>1000:
        return dict(pair_index=pair_index,
                    isomorphic=f"too large")
    t0 = time.perf_counter()
    try:
        iso, log_str = is_isomorphic(G1, G2)
        timed_out = False
    except MemoryError:
        return dict(pair_index=pair_index,
                    isomorphic=f"memory error")
    finally:
        import gc; gc.collect()

    elapsed = time.perf_counter() - t0
    print(f"[{pair_index:04}] isomorphic={iso} time={elapsed:.2f}s", flush=True)
    print(log_str)

    t1 = time.perf_counter()
    try:
        nx_iso = nx.is_isomorphic(G1, G2)
    except MemoryError:
        return dict(pair_index=pair_index,
                    isomorphic=f"memory error")
    finally:
        import gc; gc.collect()
    nx_elapsed = time.perf_counter() - t1
    print(f"[{pair_index:04}] nx.is_isomorphic={nx_iso} time={nx_elapsed:.2f}s", flush=True)

    return dict(pair_index=pair_index,
                isomorphic=iso,
                time_s=round(elapsed, 2),
                nx_isomorphic=nx_iso,
                nx_time_s=round(nx_elapsed, 2),)

def existing_pair_indices(csv_path):
    """Read previous results and return a set of completed pair indices."""
    if not csv_path.exists():
        return set()
    indices = set()
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["pair_index"])
                indices.add(idx)
            except Exception:
                continue
    return indices

def find_pairs():
    all_files = list(DATA_DIR.iterdir())
    pattern = re.compile(r"(\d+)_([12])$")
    groups = {}
    for p in all_files:
        if p.is_file():
            m = pattern.fullmatch(p.stem)
            if m:
                idx, v = m.groups()
                groups.setdefault(idx, {})[v] = p
    pairs = [(int(idx), d["1"], d["2"])
             for idx, d in groups.items() if {"1", "2"} <= d.keys()]
    pairs.sort(key=lambda x: x[0])  # sort by pair index
    return pairs

def main():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs()
    print(f"Found {len(pairs)} pairs to test. Output: {OUT_LOG}")

    done_indices = existing_pair_indices(OUT_LOG)
    print(f"{len(done_indices)} pairs already completed. Skipping these.")

    todo_pairs = [(i, f1, f2) for i, f1, f2 in pairs if i not in done_indices]
    print(f"{len(todo_pairs)} pairs to run in this round. Running with {MAX_WORKERS} workers.")

    if not todo_pairs:
        print("No new pairs to run. Exiting.")
        return

    # Append mode (create header if missing)
    need_header = not OUT_LOG.exists() or OUT_LOG.stat().st_size == 0
    fout = OUT_LOG.open("a", newline="")
    fieldnames = ["pair_index", "file1", "file2", "isomorphic",
                  "time_s", "timed_out",
                  "nx_isomorphic", "nx_time_s", "nx_error",
                  "skipped", "error"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
    if need_header:
        writer.writeheader()
        fout.flush()

    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS,initializer=impose_limits) as pool:
        futures = {}
        for i, f1, f2 in todo_pairs:
            futures[pool.submit(run_pair, i, f1, f2)] = i
            print(f"pair {i} submitted")

        done_cnt = 0
        for fut in cf.as_completed(futures):
            i = futures[fut]
            try:
                result = fut.result(timeout=TIMEOUT)
            except cf.TimeoutError:
                result = dict(pair_index=i,
                              isomorphic="timeout", timed_out="timeout")
                print(f"[{i:04}] TIMEOUT", flush=True)
            except MemoryError:
                result = dict(pair_index=i,
                              isomorphic="memory exceed")
            except Exception as exc:
                result = dict(pair_index=i,
                              isomorphic="crashed")
                print(f"[{i:04}] CRASHED", flush=True)
            writer.writerow(result)
            fout.flush()
            done_cnt += 1
            if done_cnt % 10 == 0 or done_cnt == len(futures):
                print(f"{done_cnt}/{len(futures)} pairs finished.", flush=True)

    fout.close()
    print("All pairs completed or skipped.")

if __name__ == "__main__":
    main()
