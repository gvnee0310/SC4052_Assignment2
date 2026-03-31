import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time


def load_graph(filepath):
    edges = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            src, dst = int(parts[0]), int(parts[1])
            edges.append((src, dst))
    
    all_nodes = set()
    for s, d in edges:
        all_nodes.add(s)
        all_nodes.add(d)
    nodes = sorted(all_nodes)
    n = len(nodes)
    
    # Create node-to-index mapping
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    return edges, nodes, n, node_to_idx


def build_transition_matrix_dense(edges, n, node_to_idx):
    M = np.zeros((n, n))
    out_degree = np.zeros(n)
    
    for src, dst in edges:
        i = node_to_idx[src]
        j = node_to_idx[dst]
        M[j][i] += 1
        out_degree[i] += 1
    
    for i in range(n):
        if out_degree[i] > 0:
            M[:, i] /= out_degree[i]
        else:
            M[:, i] = 1.0 / n
    
    return M


def build_transition_matrix_sparse(edges, n, node_to_idx):
    rows, cols, data = [], [], []
    out_degree = np.zeros(n)
    
    for src, dst in edges:
        i = node_to_idx[src]
        j = node_to_idx[dst]
        rows.append(j)
        cols.append(i)
        data.append(1.0)
        out_degree[i] += 1
    
    M = sparse.csc_matrix((data, (rows, cols)), shape=(n, n))
    
    for i in range(n):
        if out_degree[i] > 0:
            start = M.indptr[i]
            end = M.indptr[i + 1]
            M.data[start:end] /= out_degree[i]
    
    dangling = (out_degree == 0)
    return M, dangling


def pagerank_power_iteration(M_dense, p, max_iter=200, tol=1e-10):
    n = M_dense.shape[0]
    pi = np.ones(n) / n  # uniform initial distribution
    history = []
    
    for iteration in range(max_iter):
        pi_new = (1 - p) * M_dense @ pi + (p / n) * np.ones(n)
        
        diff = np.sum(np.abs(pi_new - pi))
        history.append(diff)
        pi = pi_new
        
        if diff < tol:
            break
    
    return pi / np.sum(pi), history, iteration + 1


def pagerank_sparse_power_iteration(M_sparse, dangling, p, n, max_iter=200, tol=1e-10):
    pi = np.ones(n) / n
    history = []
    
    for iteration in range(max_iter):
        # Dangling node contribution
        dangling_sum = np.sum(pi[dangling])
        
        # Sparse matrix-vector multiply + teleportation
        pi_new = (1 - p) * M_sparse.dot(pi) \
                 + (1 - p) * dangling_sum / n * np.ones(n) \
                 + p / n * np.ones(n)
        
        diff = np.sum(np.abs(pi_new - pi))
        history.append(diff)
        pi = pi_new
        
        if diff < tol:
            break
    
    return pi / np.sum(pi), history, iteration + 1


def pagerank_closed_form(M_dense, p):
    n = M_dense.shape[0]
    I = np.eye(n)
    
    # Solve (I - (1-p)*M) * pi = (p/n) * ones
    A = I - (1 - p) * M_dense
    b = (p / n) * np.ones(n)
    
    pi = np.linalg.solve(A, b)
    return pi / np.sum(pi)


def pagerank_closed_form_sparse(M_sparse, dangling, p, n):
    I_sparse = sparse.eye(n, format='csc')
    d = dangling.astype(float)
    
    A = I_sparse - (1 - p) * M_sparse
    b = (p / n) * np.ones(n)
    
    # Iterative approach to find consistent s
    s = 0.0  
    for _ in range(50):
        rhs = ((1 - p) * s / n + p / n) * np.ones(n)
        pi = spsolve(A, rhs)
        s_new = np.dot(d, pi)
        if abs(s_new - s) < 1e-12:
            break
        s = s_new
    
    return pi / np.sum(pi)


def compare_methods(M_dense, p):
    print(f"\n--- Comparing methods for p = {p:.2f} ---")
    
    t0 = time.time()
    pi_power, history, iters = pagerank_power_iteration(M_dense, p)
    t_power = time.time() - t0
    
    t0 = time.time()
    pi_closed = pagerank_closed_form(M_dense, p)
    t_closed = time.time() - t0
    
    diff = np.max(np.abs(pi_power - pi_closed))
    
    print(f"  Power iteration: {iters} iterations, {t_power:.4f}s")
    print(f"  Closed form:     {t_closed:.4f}s")
    print(f"  Max difference:  {diff:.2e}")
    print(f"  Top 5 nodes (power): {np.argsort(-pi_power)[:5]}")
    print(f"  Top 5 nodes (closed): {np.argsort(-pi_closed)[:5]}")
    
    return pi_power, pi_closed, history, iters


if __name__ == "__main__":
    import os
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    # Test on small example
    small_file = os.path.join(data_dir, "web-Small_6.txt")
    if os.path.exists(small_file):
        print("=" * 60)
        print("Small Example (6 nodes)")
        print("=" * 60)
        edges, nodes, n, node_to_idx = load_graph(small_file)
        M = build_transition_matrix_dense(edges, n, node_to_idx)
        print(f"Nodes: {nodes}, Edges: {len(edges)}")
        print(f"Transition matrix M:\n{np.round(M, 3)}")
        
        for p in [0.05, 0.15, 0.30, 0.50, 0.85]:
            pi_power, pi_closed, history, iters = compare_methods(M, p)
            print(f"  PageRank vector: {np.round(pi_power, 4)}")
    
    # Test on 10k graph
    large_file = os.path.join(data_dir, "web-Synthetic_10k.txt")
    if os.path.exists(large_file):
        print("\n" + "=" * 60)
        print("Large Graph (10k nodes)")
        print("=" * 60)
        edges, nodes, n, node_to_idx = load_graph(large_file)
        print(f"Nodes: {n}, Edges: {len(edges)}")
        
        # Use sparse for large graph
        M_sparse, dangling = build_transition_matrix_sparse(edges, n, node_to_idx)
        print(f"Dangling nodes: {np.sum(dangling)}")
        
        for p in [0.15]:
            t0 = time.time()
            pi_sparse, hist_sparse, iters_sparse = pagerank_sparse_power_iteration(
                M_sparse, dangling, p, n)
            t_sparse = time.time() - t0
            print(f"\n  Sparse power iteration (p={p}):")
            print(f"    {iters_sparse} iterations, {t_sparse:.4f}s")
            print(f"    Top 10 nodes: {np.argsort(-pi_sparse)[:10]}")
            print(f"    Top 10 scores: {np.round(pi_sparse[np.argsort(-pi_sparse)[:10]], 6)}")
