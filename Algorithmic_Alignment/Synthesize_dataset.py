import random
import networkx as nx
import math
from typing import List, Tuple
from Dataset import create_data
import torch

Pair = Tuple[nx.Graph, nx.Graph, bool]

def make_isomorphic_copy(G: nx.Graph) -> nx.Graph:
    """Return a randomly relabeled copy of G (guaranteed isomorphic)."""
    nodes = list(G.nodes())
    shuffled = random.sample(nodes, len(nodes))
    mapping = dict(zip(nodes, shuffled))
    return nx.relabel_nodes(G, mapping)


def make_nonisomorphic(G: nx.Graph, m: int) -> nx.Graph:
    """
    Sample graphs with same (|V|,|E|) until one is NOT isomorphic to G.
    Expected retries is tiny because random graphs are rarely isomorphic.
    """
    n = G.number_of_nodes()
    while True:
        H = nx.gnm_random_graph(n, m)  # fix |V| and |E|
        if not nx.is_isomorphic(G, H):
            return H

def make_nonisomorphic_SR(G: nx.Graph, m: int) -> nx.Graph:
    n = G.number_of_nodes()
    H = nx.random_regular_graph(m, n)
    return H


def sample_pair(iso_prob: float = 0.5) -> Pair:
    """Produce one graph pair."""
    n = random.randint(10, 250)
    p = random.uniform(0.2, 0.8)
    G1 = nx.gnp_random_graph(n, p)
    m = G1.number_of_edges()

    if random.random() < iso_prob:
        G2 = make_isomorphic_copy(G1)
        return G1, G2, True
    else:
        G2 = make_nonisomorphic(G1, m)
        return G1, G2, False


def sample_pair_SR(iso_prob: float = 0.5) -> Pair:
    """Produce one graph pair."""
    n = random.randint(2, 62)

    q = 4 * n + 1
    quadratic = {pow(i, 2, q) for i in range(1, q)}

    G1 = nx.Graph()
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in quadratic:
                G1.add_edge(i, j)

    if random.random() < iso_prob:
        G2 = make_isomorphic_copy(G1)
        return G1, G2, True
    else:
        G2 = make_nonisomorphic_SR(G1, (q-1)//2)
        return G1, G2, False

import networkx as nx, random
from typing import Tuple

def _cfi_graph(base: nx.Graph, twist: bool = False) -> nx.Graph:
    """
    Return the 'even' (twist = False) or 'odd' (twist = True)
    CFI graph built on a 3-regular base graph.

    All graphs produced share:
        |V| = 4·|V(base)|          (≤ 96 if |V(base)| ≤ 24)
        |E| = 9·|V(base)|
    Even/odd versions are non-isomorphic but have identical statistics.
    """
    H = nx.Graph()

    # one central node per base vertex
    for v in base.nodes():
        H.add_node(f"{v}_c")

    # gadget for every base edge {u,v}
    for u, v in base.edges():
        # four "half-edge" nodes
        for s in (f"{u}|{v}_0", f"{u}|{v}_1", f"{v}|{u}_0", f"{v}|{u}_1"):
            H.add_node(s)

        # connect central vertices to their half-edge nodes
        H.add_edges_from([
            (f"{u}_c", f"{u}|{v}_0"), (f"{u}_c", f"{u}|{v}_1"),
            (f"{v}_c", f"{v}|{u}_0"), (f"{v}_c", f"{v}|{u}_1")
        ])

        # even wiring (parallel) versus odd wiring (crossed)
        if twist:   # odd
            H.add_edge(f"{u}|{v}_0", f"{v}|{u}_1")
            H.add_edge(f"{u}|{v}_1", f"{v}|{u}_0")
        else:       # even
            H.add_edge(f"{u}|{v}_0", f"{v}|{u}_0")
            H.add_edge(f"{u}|{v}_1", f"{v}|{u}_1")

    return H


def sample_pair_cfi(iso: bool = False
                    ) -> Tuple[nx.Graph, nx.Graph, bool]:
    """
    n_base         # vertices in the 3-regular base graph

    """
    # 1) connected 3-regular base graph
    n_base = random.randint(5,35)
    if n_base % 2 == 1:
        n_base -= 1
    while True:
        base = nx.random_regular_graph(3, n_base)
        if nx.is_connected(base):
            break

    even = _cfi_graph(base, twist=False)

    if iso:                          # easy (isomorphic) case
        perm = dict(zip(even.nodes(), random.sample(even.nodes(), even.number_of_nodes())))
        return even, nx.relabel_nodes(even, perm), True
    else:                            # hard non-iso case
        odd = _cfi_graph(base, twist=True)
        return even, odd, False

def random_3xor_formula(n_vars: int, n_clauses: int) -> List[Tuple[int,int,int,int]]:
    """
    Each clause is (i, j, k, b) meaning x_i ⊕ x_j ⊕ x_k = b  (b ∈ {0,1})
    Variable indices start at 0.
    """
    vars_ = list(range(n_vars))
    clauses = []
    for _ in range(n_clauses):
        i, j, k = random.sample(vars_, 3)
        b       = random.randint(0, 1)
        clauses.append((i, j, k, b))
    return clauses


def three_xor_to_graph(clauses: List[Tuple[int,int,int,int]]) -> nx.Graph:
    """
    Implements the bipartite-plus-gadgets construction in §3 of Dawar & Khan (2018).
    • For each variable x  : add two literal vertices  x  and ¬x plus an edge (x,¬x).
    • For each clause C : add a ‘clause’ vertex C and connect it to the three
      (possibly negated) literals that occur in the clause.
    • Finally, make the graph 3-regular by subdividing degree-2 edges (optional).
    The resulting graph has no non-trivial automorphisms w.h.p.
    """
    g = nx.Graph()
    n_vars = max(max(i,j,k) for i,j,k,_ in clauses) + 1
    # literals
    for v in range(n_vars):
        g.add_edge(f'x{v}', f'¬x{v}')
    # clauses
    for idx,(i,j,k,b) in enumerate(clauses):
        c_name = f'C{idx}_{b}'
        for lit,neg in [(i,0),(j,0),(k,b)]:       # enforce parity gadget
            g.add_edge(c_name, f'x{lit}' if neg==0 else f'¬x{lit}')
    return g


def sample_pair_xor(iso:bool=False):
    """
    Returns (G1,G2, label) where label = 1 if isomorphic else 0.
    • If make_iso=True we simply relabel G1 to obtain G2.
    • Otherwise we flip the parity bit of *one* clause → produces non-iso pair.
    """
    n_vars: int = random.randint(10,100)
    n_clauses: int = random.randint(15,150)
    phi = random_3xor_formula(n_vars, n_clauses)
    G1  = three_xor_to_graph(phi)
    if iso:
        perm = {v: f'p{idx}' for idx,v in enumerate(G1.nodes)}
        G2   = nx.relabel_nodes(G1, perm)
        return G1, G2, 1
    else:                         # non-isomorphic: flip a clause’s RHS
        i = random.randrange(len(phi))
        i_,j_,k_,b = phi[i]
        phi[i] = (i_,j_,k_,1-b)
        G2 = three_xor_to_graph(phi)
        return G1, G2, 0


def _expand_edge_path(G_in: nx.Graph, k: int) -> nx.Graph:
    """Return a new graph where every edge is replaced by a length-(k+1) path."""
    G_out = nx.Graph()
    G_out.add_nodes_from(G_in.nodes())           # keep original labels

    next_id = max(G_in.nodes()) + 1
    for u, v in G_in.edges():
        last = u
        for _ in range(k):
            w = next_id
            next_id += 1
            G_out.add_edge(last, w)
            last = w
        G_out.add_edge(last, v)
    return G_out

def sample_pair_exp(
        make_iso: bool | None = None,
) -> Tuple[nx.Graph, nx.Graph, int]:
    """
    make_iso : bool | None

    N = n_base + k*d*n_base/2
    """
    n_base: int = random.randint(10,20)
    d: int = random.randint(3,4)
    k: int = random.randint(4,5)
    if make_iso is None:
        make_iso = bool(random.getrandbits(1))
    while (n_base * d) % 2 == 1:
        n_base = random.randint(10, 30)

    base = nx.random_regular_graph(d, n_base)
    G1   = _expand_edge_path(base, k)

    if make_iso:
        # simple isomorphic copy via node relabelling
        perm = {v: f"p{v}" for v in G1.nodes()}
        G2   = nx.relabel_nodes(G1, perm)
        label = 1
    else:
        # make a *non*-iso mate: flip one gadget path (reverse direction)
        G2 = G1.copy()
        # pick an original edge and reverse its gadget path
        for u, v in base.edges():
            path_nodes = [n for n in nx.shortest_path(G2, u, v)][1:-1]
            if path_nodes:
                G2.remove_edges_from(nx.utils.pairwise([u, *path_nodes, v]))
                G2.add_edges_from(nx.utils.pairwise([v, *reversed(path_nodes), u]))
                break
        label = 0

    return G1, G2, label

if __name__ == "__main__":
    num_data = 12
    for data_name in ["syn","sr","cfi","3xor","exp"]:
        g_list, y_list = [], []
        if data_name == "syn":
            for i in range(num_data):
                iso = 1.0 if i < num_data // 2 else 0.0
                G1, G2, _ = sample_pair(iso)
                g_list.append((create_data(G1), create_data(G2)))
                y_list.append(int(iso))
            for i, (g1, g2) in enumerate(g_list):
                n = g1.x.shape[0]
                torch.save(g1, f"{data_name}/{data_name}_graphs_{i+500}_1_{n}_{y_list[i]}.pt")
                torch.save(g2, f"{data_name}/{data_name}_graphs_{i+500}_2_{n}_{y_list[i]}.pt")
            print(f"{data_name} stored")
        elif data_name == "sr":
            for i in range(num_data):
                iso = 1.0 if i < num_data // 2 else 0.0
                G1, G2, _ = sample_pair_SR(iso)
                g_list.append((create_data(G1), create_data(G2)))
                y_list.append(int(iso))
            for i, (g1, g2) in enumerate(g_list):
                n = g1.x.shape[0]
                torch.save(g1, f"{data_name}/{data_name}_graphs_{i+500}_1_{n}_{y_list[i]}.pt")
                torch.save(g2, f"{data_name}/{data_name}_graphs_{i+500}_2_{n}_{y_list[i]}.pt")
            print(f"{data_name} stored")
        elif data_name == "cfi":
            for i in range(num_data):
                iso = 1.0 if i < num_data // 2 else 0.0
                G1, G2, _ = sample_pair_cfi(iso)
                g_list.append((create_data(G1), create_data(G2)))
                y_list.append(int(iso))
            for i, (g1, g2) in enumerate(g_list):
                n = g1.x.shape[0]
                torch.save(g1, f"{data_name}/{data_name}_graphs_{i+500}_1_{n}_{y_list[i]}.pt")
                torch.save(g2, f"{data_name}/{data_name}_graphs_{i+500}_2_{n}_{y_list[i]}.pt")
            print(f"{data_name} stored")
        elif data_name == "3xor":
            for i in range(num_data):
                iso = 1.0 if i < num_data // 2 else 0.0
                G1, G2, _ = sample_pair_xor(iso)
                g_list.append((create_data(G1), create_data(G2)))
                y_list.append(int(iso))
            for i, (g1, g2) in enumerate(g_list):
                n = g1.x.shape[0]
                torch.save(g1, f"{data_name}/{data_name}_graphs_{i+500}_1_{n}_{y_list[i]}.pt")
                torch.save(g2, f"{data_name}/{data_name}_graphs_{i+500}_2_{n}_{y_list[i]}.pt")
            print(f"{data_name} stored")
        elif data_name == "exp":
            for i in range(num_data):
                iso = 1.0 if i < num_data // 2 else 0.0
                G1, G2, _ = sample_pair_exp(iso)
                g_list.append((create_data(G1), create_data(G2)))
                y_list.append(int(iso))
            for i, (g1, g2) in enumerate(g_list):
                n = g1.x.shape[0]
                torch.save(g1, f"{data_name}/{data_name}_graphs_{i+500}_1_{n}_{y_list[i]}.pt")
                torch.save(g2, f"{data_name}/{data_name}_graphs_{i+500}_2_{n}_{y_list[i]}.pt")
            print(f"{data_name} stored")