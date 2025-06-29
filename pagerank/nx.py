import networkx as nx

# Step 1: 读取边列表并构建有向图
def build_graph(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                from_node, to_node = map(int, line.strip().split())
                G.add_edge(from_node, to_node)
    return G

# Step 2: 计算PageRank
def compute_pagerank(G, alpha=0.85, tol=1e-10, max_iter=1000):
    # 明确dangling节点处理方式 — 默认均匀分布（networkx默认即如此）
    n = G.number_of_nodes()
    personalization = {node: 1 / n for node in G.nodes()}
    
    pr = nx.pagerank(
        G, 
        alpha=alpha, 
        personalization=personalization,
        tol=tol, 
        max_iter=max_iter
    )
    return pr

# Step 3: 保存结果Top100
def save_topk(pr_scores, filename, k=100):
    sorted_items = sorted(pr_scores.items(), key=lambda x: -x[1])[:k]
    with open(filename, 'w') as f:
        for node_id, score in sorted_items:
            f.write(f"{node_id} {score:.8f}\n")

if __name__ == "__main__":
    G = build_graph('Data.txt')
    pr_scores = compute_pagerank(G, alpha=0.85, tol=1e-10, max_iter=1000)
    save_topk(pr_scores, 'Res_nx.txt', k=100)
    print("networkx PageRank (high precision) results saved to Res_nx.txt")
