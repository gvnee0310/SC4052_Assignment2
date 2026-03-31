import numpy as np
import os
import sys
import re
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROBOTS_TXT_DB = {
    "https://www.example-university.edu": {
        "User-agent": "*",
        "Allow": "/",
        "Disallow": "/private/",
        # GPTBot not explicitly blocked -> allowed
    },
    "https://news.example.com": {
        "User-agent": "GPTBot",
        "Disallow": "/",  # Blocks AI crawlers entirely
    },
    "https://research.example.org": {
        "User-agent": "*",
        "Allow": "/",
        "User-agent-GPTBot": "Allow: /papers/",
    },
    "https://blog.tech-company.com": {
        "User-agent": "*",
        "Allow": "/",
    },
    "https://wiki.example.org": {
        "User-agent": "*",
        "Allow": "/",
    },
    "https://govt-data.example.gov": {
        "User-agent": "*",
        "Allow": "/",
    },
    "https://social-media.example.com": {
        "User-agent": "GPTBot",
        "Disallow": "/",  # Blocks AI crawlers
    },
    "https://docs.ai-framework.dev": {
        "User-agent": "*",
        "Allow": "/",
    },
}


def check_robots_txt(url, user_agent="GPTBot"):
    """Check if a URL permits crawling by the specified user-agent.
    
    Simplified robots.txt parser. In production, use robotparser from urllib.
    
    Rules:
    1. If GPTBot is explicitly disallowed -> blocked
    2. If wildcard (*) disallows the path -> blocked
    3. Otherwise -> allowed (permissive default)
    """
    domain = urlparse(url).scheme + "://" + urlparse(url).netloc
    
    if domain in ROBOTS_TXT_DB:
        rules = ROBOTS_TXT_DB[domain]
        
        # Check GPTBot-specific rules first
        if rules.get("User-agent") == "GPTBot" and rules.get("Disallow") == "/":
            return False
        if f"User-agent-{user_agent}" in rules:
            # Has specific GPTBot rules
            return True  # Simplified: assume specific rules are permissive
        
        # Check wildcard rules
        path = urlparse(url).path
        disallow = rules.get("Disallow", "")
        if disallow and disallow != "/" and path.startswith(disallow):
            return False
        if disallow == "/" and rules.get("User-agent") == "GPTBot":
            return False
    
    # Default: allow (no robots.txt found = permissive)
    return True


# ============================================================
# Content quality heuristics
# ============================================================

def content_quality_signal(url):
    """Estimate content quality from URL structure.
    
    Higher scores for:
    - .edu, .gov, .org domains (institutional authority)
    - /research/, /papers/, /docs/ paths (academic/technical content)
    - /api/, /documentation/ paths (technical reference)
    
    Lower scores for:
    - /social/, /feed/, /trending/ paths (social/viral content)
    - Very deep URL paths (often low-quality auto-generated pages)
    """
    score = 0.5  # baseline
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # Domain authority signals
    if domain.endswith('.edu'):
        score += 0.3
    elif domain.endswith('.gov'):
        score += 0.25
    elif domain.endswith('.org'):
        score += 0.15
    elif domain.endswith('.dev'):
        score += 0.1
    
    # Path quality signals
    high_quality_paths = ['/research/', '/papers/', '/docs/', '/documentation/',
                          '/api/', '/publications/', '/reports/', '/data/']
    for hq_path in high_quality_paths:
        if hq_path in path:
            score += 0.2
            break
    
    low_quality_paths = ['/social/', '/feed/', '/trending/', '/ads/',
                         '/tracking/', '/redirect/']
    for lq_path in low_quality_paths:
        if lq_path in path:
            score -= 0.2
            break
    
    # Depth penalty: very deep URLs are often low quality
    depth = path.count('/')
    if depth > 5:
        score -= 0.1 * (depth - 5)
    
    return np.clip(score, 0.0, 1.0)


def freshness_signal(url):
    """Estimate content freshness from URL patterns.
    
    In production, this would use HTTP headers (Last-Modified, ETag).
    Here we use URL-based heuristics as a proxy.
    """
    path = urlparse(url).path.lower()
    
    # Look for date patterns in URL
    year_pattern = re.findall(r'/20[12]\d/', path)
    if year_pattern:
        year = int(year_pattern[-1].strip('/'))
        if year >= 2023:
            return 0.9
        elif year >= 2020:
            return 0.6
        else:
            return 0.3
    
    # Default moderate freshness
    return 0.5


# ============================================================
# Crawl prioritization engine
# ============================================================

def compute_crawl_priority(web_graph, pagerank_scores, k=10,
                           w_pagerank=0.5, w_content=0.3, w_freshness=0.2):
    """Compute crawl priority for all URLs and return top-k.
    
    Args:
        web_graph: dict mapping URL -> list of outlink URLs
        pagerank_scores: dict mapping URL -> PageRank score
        k: number of top URLs to return
        w_pagerank: weight for PageRank component
        w_content: weight for content quality component
        w_freshness: weight for freshness component
    
    Returns:
        ranked_urls: list of (url, composite_score, components_dict) tuples
        blocked_urls: list of URLs blocked by robots.txt
    """
    all_urls = set(web_graph.keys())
    for outlinks in web_graph.values():
        all_urls.update(outlinks)
    
    # Normalize PageRank scores to [0, 1]
    pr_values = [pagerank_scores.get(url, 0) for url in all_urls]
    pr_max = max(pr_values) if pr_values else 1
    pr_min = min(pr_values) if pr_values else 0
    pr_range = pr_max - pr_min if pr_max > pr_min else 1
    
    results = []
    blocked = []
    
    for url in all_urls:
        # Check robots.txt
        if not check_robots_txt(url):
            blocked.append(url)
            continue
        
        # Compute components
        pr_norm = (pagerank_scores.get(url, 0) - pr_min) / pr_range
        cq = content_quality_signal(url)
        fs = freshness_signal(url)
        
        # Composite score (AACS)
        composite = w_pagerank * pr_norm + w_content * cq + w_freshness * fs
        
        results.append((url, composite, {
            'pagerank': pagerank_scores.get(url, 0),
            'pagerank_normalized': pr_norm,
            'content_quality': cq,
            'freshness': fs,
            'robots_allowed': True,
        }))
    
    # Sort by composite score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:k], blocked


# ============================================================
# Demo: Sample web graph
# ============================================================

def create_sample_web_graph():
    """Create a realistic small web graph for demonstration."""
    web_graph = {
        "https://www.example-university.edu/research/ai/": [
            "https://www.example-university.edu/research/ai/papers/",
            "https://docs.ai-framework.dev/tutorials/",
            "https://blog.tech-company.com/ai-trends-2024/",
        ],
        "https://www.example-university.edu/research/ai/papers/": [
            "https://www.example-university.edu/research/ai/",
            "https://research.example.org/papers/ml-survey/",
        ],
        "https://news.example.com/tech/ai-regulation/": [
            "https://govt-data.example.gov/reports/ai-policy/",
            "https://blog.tech-company.com/ai-trends-2024/",
        ],
        "https://blog.tech-company.com/ai-trends-2024/": [
            "https://docs.ai-framework.dev/tutorials/",
            "https://www.example-university.edu/research/ai/",
            "https://social-media.example.com/trending/ai/",
        ],
        "https://research.example.org/papers/ml-survey/": [
            "https://www.example-university.edu/research/ai/papers/",
            "https://wiki.example.org/machine-learning/",
        ],
        "https://wiki.example.org/machine-learning/": [
            "https://www.example-university.edu/research/ai/",
            "https://docs.ai-framework.dev/tutorials/",
            "https://research.example.org/papers/ml-survey/",
        ],
        "https://docs.ai-framework.dev/tutorials/": [
            "https://docs.ai-framework.dev/api/reference/",
            "https://blog.tech-company.com/ai-trends-2024/",
        ],
        "https://docs.ai-framework.dev/api/reference/": [
            "https://docs.ai-framework.dev/tutorials/",
        ],
        "https://govt-data.example.gov/reports/ai-policy/": [
            "https://www.example-university.edu/research/ai/",
            "https://news.example.com/tech/ai-regulation/",
        ],
        "https://social-media.example.com/trending/ai/": [
            "https://news.example.com/tech/ai-regulation/",
            "https://blog.tech-company.com/ai-trends-2024/",
        ],
    }
    return web_graph


def compute_pagerank_for_url_graph(web_graph, p=0.15):
    """Compute PageRank for a URL-based web graph."""
    # Map URLs to indices
    all_urls = set(web_graph.keys())
    for outlinks in web_graph.values():
        all_urls.update(outlinks)
    url_list = sorted(all_urls)
    url_to_idx = {url: i for i, url in enumerate(url_list)}
    n = len(url_list)
    
    # Build transition matrix
    M = np.zeros((n, n))
    for src, outlinks in web_graph.items():
        i = url_to_idx[src]
        if outlinks:
            for dst in outlinks:
                j = url_to_idx[dst]
                M[j][i] += 1
            M[:, i] /= len(outlinks)
        else:
            M[:, i] = 1.0 / n
    
    # Handle dangling nodes (URLs that appear only as targets)
    for url in all_urls:
        idx = url_to_idx[url]
        if url not in web_graph or not web_graph.get(url):
            M[:, idx] = 1.0 / n
    
    # Power iteration
    pi = np.ones(n) / n
    for _ in range(200):
        pi_new = (1 - p) * M @ pi + (p / n) * np.ones(n)
        if np.sum(np.abs(pi_new - pi)) < 1e-10:
            break
        pi = pi_new
    pi = pi / np.sum(pi)
    
    return {url: pi[url_to_idx[url]] for url in url_list}


def visualize_crawl_results(ranked, blocked, figure_dir):
    """Generate visualization of crawl prioritization results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of top URLs by composite score
    urls_short = [urlparse(r[0]).netloc + urlparse(r[0]).path[:25] for r in ranked]
    scores = [r[1] for r in ranked]
    pr_components = [r[2]['pagerank_normalized'] * 0.5 for r in ranked]
    cq_components = [r[2]['content_quality'] * 0.3 for r in ranked]
    fs_components = [r[2]['freshness'] * 0.2 for r in ranked]
    
    y_pos = np.arange(len(ranked))
    
    ax1.barh(y_pos, pr_components, color='#2196F3', label='PageRank (50%)', edgecolor='white')
    ax1.barh(y_pos, cq_components, left=pr_components, color='#4CAF50', 
             label='Content Quality (30%)', edgecolor='white')
    left_2 = [a + b for a, b in zip(pr_components, cq_components)]
    ax1.barh(y_pos, fs_components, left=left_2, color='#FF9800', 
             label='Freshness (20%)', edgecolor='white')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(urls_short, fontsize=7)
    ax1.set_xlabel('Composite Score (AACS)')
    ax1.set_title('Top URLs to Crawl — Authority-Accessible Composite Score')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Pie chart: allowed vs blocked
    n_allowed = len(ranked)
    n_blocked = len(blocked)
    ax2.pie([n_allowed, n_blocked], labels=['Crawlable', 'Blocked by robots.txt'],
            colors=['#4CAF50', '#F44336'], autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 11})
    ax2.set_title('robots.txt Compliance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "fig8_crawl_priority.png"))
    plt.close()


def print_crawl_report(ranked, blocked):
    """Print a detailed crawl prioritization report."""
    print("\n" + "=" * 80)
    print("AI WEB CRAWLING STRATEGY — PRIORITIZATION REPORT")
    print("=" * 80)
    
    print(f"\n{'Rank':<5} {'Composite':<10} {'PageRank':<10} {'Quality':<10} {'Fresh':<8} URL")
    print("-" * 80)
    
    for i, (url, score, components) in enumerate(ranked, 1):
        print(f"{i:<5} {score:<10.4f} {components['pagerank']:<10.6f} "
              f"{components['content_quality']:<10.2f} {components['freshness']:<8.2f} "
              f"{url[:50]}")
    
    if blocked:
        print(f"\nBlocked by robots.txt ({len(blocked)} URLs):")
        for url in blocked:
            print(f"  ✗ {url}")
            

if __name__ == "__main__":
    FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    print("Creating sample web graph...")
    web_graph = create_sample_web_graph()
    
    print("Computing PageRank scores...")
    pr_scores = compute_pagerank_for_url_graph(web_graph, p=0.15)
    
    print("\nPageRank scores:")
    for url, score in sorted(pr_scores.items(), key=lambda x: -x[1]):
        print(f"  {score:.6f}  {url}")
    
    print("\nRunning crawl prioritization (top 8)...")
    ranked, blocked = compute_crawl_priority(web_graph, pr_scores, k=8)
    
    print_crawl_report(ranked, blocked)
    visualize_crawl_results(ranked, blocked, FIGURE_DIR)
    print(f"\nCrawl priority figure saved to {FIGURE_DIR}")
