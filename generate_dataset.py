import numpy as np
import os

def generate_power_law_web_graph(n_nodes=10000, avg_out_degree=8, seed=42):
    rng = np.random.default_rng(seed)
    edges = set()
    
    # Power-law out-degree distribution
    out_degrees = np.maximum(1, (rng.pareto(1.5, n_nodes) * avg_out_degree / 2).astype(int))
    out_degrees = np.minimum(out_degrees, n_nodes - 1)
    
    for src in range(n_nodes):
        # Preferential attachment: higher-index nodes link to lower-index (authority) nodes more
        n_links = out_degrees[src]
        # Mix of uniform random + preferential attachment
        targets = set()
        while len(targets) < n_links:
            if rng.random() < 0.6:
                # Preferential: bias toward nodes with lower IDs (simulating popular pages)
                t = int(rng.exponential(n_nodes * 0.3)) % n_nodes
            else:
                t = rng.integers(0, n_nodes)
            if t != src:
                targets.add(t)
        for t in targets:
            edges.add((src, t))
    
    return sorted(edges)

def save_edge_list(edges, filepath, n_nodes):
    with open(filepath, 'w') as f:
        f.write(f"# Directed graph (synthetic): Nodes: {n_nodes} Edges: {len(edges)}\n")
        f.write("# FromNodeId\tToNodeId\n")
        for src, dst in edges:
            f.write(f"{src}\t{dst}\n")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(out_dir), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate 10k-node graph
    print("Generating 10k-node synthetic web graph...")
    edges = generate_power_law_web_graph(n_nodes=10000, avg_out_degree=8, seed=42)
    save_edge_list(edges, os.path.join(data_dir, "web-Synthetic_10k.txt"), 10000)
    print(f"  Saved {len(edges)} edges")

    # Generate small illustrative example (6 nodes)
    small_edges = [
        (0, 1), (0, 2),       # Node 0 -> 1, 2
        (1, 2),               # Node 1 -> 2
        (2, 0), (2, 3),      # Node 2 -> 0, 3
        (3, 4), (3, 5),      # Node 3 -> 4, 5
        (4, 5),              # Node 4 -> 5
        (5, 0),              # Node 5 -> 0 (back link)
    ]
    save_edge_list(small_edges, os.path.join(data_dir, "web-Small_6.txt"), 6)
    print(f"  Saved small example with {len(small_edges)} edges")
    print("Done.")
