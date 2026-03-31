import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pagerank_core import (
    load_graph, build_transition_matrix_dense,
    build_transition_matrix_sparse,
    pagerank_power_iteration, pagerank_closed_form,
    pagerank_sparse_power_iteration
)

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'savefig.dpi': 180,
    'savefig.bbox': 'tight',
})


def fig1_small_graph():
    print("Figure 1: Small graph visualization...")
    
    edges_list = [
        (0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 4), (3, 5), (4, 5), (5, 0)
    ]
    G = nx.DiGraph(edges_list)
    
    edges, nodes, n, node_to_idx = load_graph(os.path.join(DATA_DIR, "web-Small_6.txt"))
    M = build_transition_matrix_dense(edges, n, node_to_idx)
    pi, _, _ = pagerank_power_iteration(M, p=0.15)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, k=2)
    
    node_sizes = [pi[i] * 8000 + 400 for i in range(n)]
    node_colors = [pi[i] for i in range(n)]
    
    drawn = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                                    node_color=node_colors, cmap=plt.cm.YlOrRd,
                                    edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', arrows=True,
                           arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.1', width=1.5)
    labels = {i: f"Node {i}\nPR={pi[i]:.3f}" for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=9, font_weight='bold')
    
    ax.set_title("6-Node Web Graph with PageRank Scores (p = 0.15)", fontweight='bold')
    ax.axis('off')
    plt.colorbar(drawn, ax=ax, label='PageRank Score', shrink=0.7)
    plt.savefig(os.path.join(FIGURE_DIR, "fig1_small_graph.png"))
    plt.close()
    print(f"  PageRank: {np.round(pi, 4)}")


def fig2_effect_of_p():
    print("Figure 2: Effect of p...")
    
    edges, nodes, n, node_to_idx = load_graph(os.path.join(DATA_DIR, "web-Small_6.txt"))
    M = build_transition_matrix_dense(edges, n, node_to_idx)
    
    p_values = np.linspace(0.01, 0.99, 50)
    pr_matrix = np.zeros((len(p_values), n))
    for idx, p in enumerate(p_values):
        pi, _, _ = pagerank_power_iteration(M, p, max_iter=500)
        pr_matrix[idx] = pi
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i in range(n):
        ax1.plot(p_values, pr_matrix[:, i], label=f'Node {i}', color=colors[i], linewidth=2)
    ax1.axhline(y=1/n, color='gray', linestyle='--', alpha=0.7, label=f'Uniform (1/{n})')
    ax1.set_xlabel('Teleportation Probability p')
    ax1.set_ylabel('PageRank Score')
    ax1.set_title('PageRank vs. Teleportation Probability p')
    ax1.legend(loc='center right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Entropy
    entropies = []
    for idx in range(len(p_values)):
        pi = pr_matrix[idx]
        pi_pos = pi[pi > 0]
        entropies.append(-np.sum(pi_pos * np.log2(pi_pos)))
    
    ax2.plot(p_values, entropies, 'b-', linewidth=2)
    ax2.axhline(y=np.log2(n), color='red', linestyle='--', alpha=0.7, label=f'Max entropy (log₂{n})')
    ax2.set_xlabel('Teleportation Probability p')
    ax2.set_ylabel('Shannon Entropy (bits)')
    ax2.set_title('Distribution Entropy vs. p')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig2_effect_of_p.png"))
    plt.close()


def fig3_convergence():
    print("Figure 3: Convergence...")
    
    edges, nodes, n, node_to_idx = load_graph(os.path.join(DATA_DIR, "web-Small_6.txt"))
    M = build_transition_matrix_dense(edges, n, node_to_idx)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in [0.05, 0.10, 0.15, 0.30, 0.50, 0.85]:
        _, history, iters = pagerank_power_iteration(M, p, max_iter=200, tol=1e-14)
        ax.semilogy(range(1, len(history)+1), history, label=f'p={p:.2f} ({iters} iters)', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L1 Norm Change (log scale)')
    ax.set_title('Convergence of Power Iteration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig3_convergence.png"))
    plt.close()


def fig4_power_vs_closed():
    print("Figure 4: Power vs closed-form...")
    
    edges, nodes, n, node_to_idx = load_graph(os.path.join(DATA_DIR, "web-Small_6.txt"))
    M = build_transition_matrix_dense(edges, n, node_to_idx)
    
    p = 0.15
    pi_power, _, _ = pagerank_power_iteration(M, p)
    pi_closed = pagerank_closed_form(M, p)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    x = np.arange(n)
    width = 0.35
    ax1.bar(x - width/2, pi_power, width, label='Power Iteration', color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, pi_closed, width, label='Closed Form', color='coral', edgecolor='black')
    ax1.set_xlabel('Node')
    ax1.set_ylabel('PageRank Score')
    ax1.set_title(f'PageRank Comparison (p = {p})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Node {i}' for i in range(n)])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Scatter
    ax2.scatter(pi_closed, pi_power, s=100, c='steelblue', edgecolors='black', zorder=5)
    lo = min(pi_power.min(), pi_closed.min()) * 0.9
    hi = max(pi_power.max(), pi_closed.max()) * 1.1
    ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='y = x')
    ax2.set_xlabel('Closed-Form PageRank')
    ax2.set_ylabel('Power Iteration PageRank')
    ax2.set_title(f'Correlation (max diff = {np.max(np.abs(pi_power-pi_closed)):.2e})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    for i in range(n):
        ax2.annotate(f'  Node {i}', (pi_closed[i], pi_power[i]), fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig4_power_vs_closed.png"))
    plt.close()


def fig5_google_graph():
    print("Figure 5: web-Google_10k.txt analysis...")
    
    filepath = os.path.join(DATA_DIR, "web-Google_10k.txt")
    edges, nodes, n, node_to_idx = load_graph(filepath)
    M_sparse, dangling = build_transition_matrix_sparse(edges, n, node_to_idx)
    
    print(f"  Nodes: {n}, Edges: {len(edges)}, Dangling: {np.sum(dangling)}")
    
    p = 0.15
    t0 = time.time()
    pi, history, iters = pagerank_sparse_power_iteration(M_sparse, dangling, p, n)
    elapsed = time.time() - t0
    
    print(f"  Converged in {iters} iterations, {elapsed:.4f}s")
    print(f"  Top 10 node IDs: {[nodes[i] for i in np.argsort(-pi)[:10]]}")
    print(f"  Top 10 scores:   {np.round(pi[np.argsort(-pi)[:10]], 6)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    ax = axes[0, 0]
    ax.hist(pi, bins=80, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('PageRank Score')
    ax.set_ylabel('Number of Nodes')
    ax.set_title(f'PageRank Distribution (n={n:,}, p={p})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Log-log rank plot
    ax = axes[0, 1]
    sorted_pr = np.sort(pi)[::-1]
    ax.loglog(np.arange(1, n+1), sorted_pr, 'b-', linewidth=1.5)
    ax.set_xlabel('Rank')
    ax.set_ylabel('PageRank Score')
    ax.set_title('Rank Distribution (log-log)')
    ax.grid(True, alpha=0.3)
    
    # Top 20 bar chart
    ax = axes[1, 0]
    top_k = 20
    top_idx = np.argsort(-pi)[:top_k]
    ax.barh(range(top_k), pi[top_idx], color='coral', edgecolor='black')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Node {nodes[i]}' for i in top_idx], fontsize=8)
    ax.set_xlabel('PageRank Score')
    ax.set_title(f'Top {top_k} Nodes by PageRank')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Convergence
    ax = axes[1, 1]
    ax.semilogy(range(1, len(history)+1), history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L1 Norm Change')
    ax.set_title(f'Convergence ({iters} iters, {elapsed:.3f}s)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('PageRank Analysis on web-Google_10k.txt (Google 2002 Dataset)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig5_google_graph.png"))
    plt.close()
    
    return pi, nodes, n


def fig6_p_effect_google(pi_ref, nodes, n):
    print("Figure 6: Effect of p on Google graph...")
    
    filepath = os.path.join(DATA_DIR, "web-Google_10k.txt")
    edges, nodes_list, n, node_to_idx = load_graph(filepath)
    M_sparse, dangling = build_transition_matrix_sparse(edges, n, node_to_idx)
    
    p_values = [0.01, 0.05, 0.10, 0.15, 0.30, 0.50, 0.85]
    top_k = 10
    ref_top = set(np.argsort(-pi_ref)[:top_k])
    
    gini_coeffs = []
    top_overlaps = []
    iters_list = []
    
    for p in p_values:
        pi, _, iters = pagerank_sparse_power_iteration(M_sparse, dangling, p, n)
        iters_list.append(iters)
        
        # Gini
        sorted_pi = np.sort(pi)
        idx = np.arange(1, n+1)
        gini = (2 * np.sum(idx * sorted_pi)) / (n * np.sum(sorted_pi)) - (n+1)/n
        gini_coeffs.append(gini)
        
        # Top-k overlap
        cur_top = set(np.argsort(-pi)[:top_k])
        top_overlaps.append(len(ref_top & cur_top) / top_k)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    
    ax1.plot(p_values, gini_coeffs, 'bo-', linewidth=2, markersize=7)
    ax1.set_xlabel('Teleportation Probability p')
    ax1.set_ylabel('Gini Coefficient')
    ax1.set_title('PageRank Inequality vs. p')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(p_values, top_overlaps, 'rs-', linewidth=2, markersize=7)
    ax2.set_xlabel('Teleportation Probability p')
    ax2.set_ylabel(f'Overlap with p=0.15 Top-{top_k}')
    ax2.set_title(f'Top-{top_k} Ranking Stability')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(p_values, iters_list, 'g^-', linewidth=2, markersize=7)
    ax3.set_xlabel('Teleportation Probability p')
    ax3.set_ylabel('Iterations to Converge')
    ax3.set_title('Convergence Speed vs. p')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig6_p_effect_google.png"))
    plt.close()


def fig7_traps_deadends():
    print("Figure 7: Spider traps & dead ends...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Dead end
    G1 = nx.DiGraph([(0,1),(1,2)])
    pr1_low = nx.pagerank(G1, alpha=0.999)
    pr1_norm = nx.pagerank(G1, alpha=0.85)
    ax = axes[0]
    pos1 = {0:(0,0), 1:(1,0), 2:(2,0)}
    nx.draw(G1, pos1, ax=ax, labels={0:'A',1:'B',2:'C\n(dead end)'},
            node_color=['#64B5F6','#64B5F6','#EF5350'], node_size=1200,
            font_size=9, font_weight='bold', arrows=True, arrowsize=25,
            edge_color='gray', width=2)
    ax.set_title(f'Dead End\np=0.15: PR(C)={pr1_norm[2]:.3f}\np≈0: PR(C)={pr1_low[2]:.3f}')
    ax.set_xlim(-0.5, 2.5)
    
    # Spider trap
    G2 = nx.DiGraph([(0,1),(1,2),(2,2)])
    pr2_low = nx.pagerank(G2, alpha=0.999)
    pr2_norm = nx.pagerank(G2, alpha=0.85)
    ax = axes[1]
    nx.draw(G2, pos1, ax=ax, labels={0:'A',1:'B',2:'C\n(trap)'},
            node_color=['#64B5F6','#64B5F6','#FF9800'], node_size=1200,
            font_size=9, font_weight='bold', arrows=True, arrowsize=25,
            edge_color='gray', width=2)
    ax.set_title(f'Spider Trap\np=0.15: PR(C)={pr2_norm[2]:.3f}\np≈0: PR(C)={pr2_low[2]:.3f}')
    ax.set_xlim(-0.5, 2.5)
    
    # Healthy
    G3 = nx.DiGraph([(0,1),(1,2),(2,0),(0,2)])
    pr3 = nx.pagerank(G3, alpha=0.85)
    ax = axes[2]
    pos3 = {0:(0,0.5),1:(1,1),2:(1,0)}
    nx.draw(G3, pos3, ax=ax,
            labels={i:f'{chr(65+i)}\n{pr3[i]:.3f}' for i in range(3)},
            node_color=['#81C784']*3, node_size=1200,
            font_size=9, font_weight='bold', arrows=True, arrowsize=25,
            edge_color='gray', width=2, connectionstyle='arc3,rad=0.15')
    ax.set_title('Healthy Graph\n(balanced distribution)')
    ax.set_xlim(-0.5, 1.5)
    
    plt.suptitle('Pathological Structures & the Role of Teleportation', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig7_traps_deadends.png"))
    plt.close()


def fig8_scalability():
    print("Figure 8: Scalability...")
    
    rng = np.random.default_rng(42)
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    times_sparse = []
    
    for sz in sizes:
        n_e = sz * 8
        src = rng.integers(0, sz, n_e)
        tgt = rng.integers(0, sz, n_e)
        edges = list(set((int(s),int(t)) for s,t in zip(src,tgt) if s!=t))
        nti = {i:i for i in range(sz)}
        M_sp, dang = build_transition_matrix_sparse(edges, sz, nti)
        t0 = time.time()
        for _ in range(3):
            pagerank_sparse_power_iteration(M_sp, dang, 0.15, sz)
        times_sparse.append((time.time()-t0)/3)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(sizes, times_sparse, 'bo-', linewidth=2, markersize=8, label='Sparse Power Iteration')
    # O(n) reference
    ref = [times_sparse[0] * s / sizes[0] for s in sizes]
    ax.loglog(sizes, ref, 'k--', alpha=0.4, label='O(n) reference')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('PageRank Scalability (Sparse Implementation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig8_scalability.png"))
    plt.close()


if __name__ == "__main__":
    small_file = os.path.join(DATA_DIR, "web-Small_6.txt")
    if not os.path.exists(small_file):
        os.system(f"cd {os.path.dirname(os.path.abspath(__file__))} && python generate_dataset.py")
    
    fig1_small_graph()
    fig2_effect_of_p()
    fig3_convergence()
    fig4_power_vs_closed()
    pi, nodes, n = fig5_google_graph()
    fig6_p_effect_google(pi, nodes, n)
    fig7_traps_deadends()
    fig8_scalability()
    print(f"\nAll figures saved to {FIGURE_DIR}")
