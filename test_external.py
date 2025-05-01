import os
import time
import csv
import queue
import networkx as nx
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from main import is_isomorphic

DATA_DIR = "./benchmark_new"
OUTPUT_LOG = "benchmark_new.csv"
TIMEOUT_S  = 300

def read_dimacs_graph(path):
    G = nx.Graph()
    num_nodes = None

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'p' and parts[1] == 'edge':
                num_nodes = int(parts[2])
            elif parts[0] == 'e':
                u, v = int(parts[1]) - 1, int(parts[2]) - 1
                G.add_edge(u, v)

    if num_nodes is None:
        raise ValueError(f"No 'p edge' line found in {path}")
    G.add_nodes_from(range(num_nodes))
    return G

def _iso_test(args):
    G1, G2 = args
    return is_isomorphic(G1, G2)

def benchmark(n_pairs=451, data_dir=DATA_DIR, log_path=OUTPUT_LOG):
    executor = ProcessPoolExecutor(max_workers=1)
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["pair_index", "isomorphic", "ground_truth", "timed_out", "runtime_s"])
        writer.writeheader()

        for i in range(n_pairs):
            f1 = os.path.join(data_dir, f"{i}_1")
            f2 = os.path.join(data_dir, f"{i}_2")
            if not os.path.exists(f1) or not os.path.exists(f2):
                print(f"Warning: skipping missing pair {i}")
                continue

            G1 = read_dimacs_graph(f1)
            G2 = read_dimacs_graph(f2)

            if G1.number_of_nodes() < 300:
                future = executor.submit(_iso_test, (G1, G2))
                t0 = time.perf_counter()
                try:
                    iso = future.result(timeout=TIMEOUT_S)
                    timed_out = False
                except TimeoutError:
                    iso = "TIMEOUT"
                    timed_out = True
                    future.cancel()
                t1 = time.perf_counter()
                truth = nx.is_isomorphic(G1, G2)
                runtime = t1 - t0
                writer.writerow({
                    "pair_index": i,
                    "isomorphic": iso,
                    "ground_truth": truth,
                    "timed_out": timed_out,
                    "runtime_s": f"{runtime:.6f}"
                })
                status = "TIMEOUT" if timed_out else f"{runtime:.3f}s"
                print(f"[{i:03d}] isomorphic={iso} ground_truth={truth} status={status}")
            else:
                writer.writerow({
                    "pair_index": i,
                    "isomorphic": "skipped",
                    "ground_truth": "skipped",
                    "timed_out": "skipped",
                    "runtime_s": "skipped"
                })
                print(f"[{i:03d}] skipped")
    print(f"\nDone. Log written to {log_path}")


if __name__ == "__main__":
    benchmark(data_dir=DATA_DIR)
