import numpy as np
import argparse
import time
def load_graph(file_path):
    """加载边列表并构建稠密邻接矩阵"""
    edges = []
    node_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            from_node, to_node = map(int, line.strip().split())
            edges.append((from_node, to_node))
            node_set.update([from_node, to_node])

    node_list = sorted(list(node_set))
    node_id_map = {node: idx for idx, node in enumerate(node_list)}

    n = len(node_list)
    adj_matrix = np.zeros((n, n), dtype=np.float64)

    for from_node, to_node in edges:
        i = node_id_map[to_node]   # PageRank是反向边
        j = node_id_map[from_node]
        adj_matrix[i, j] = 1.0

    return adj_matrix, node_list, node_id_map


def pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """稠密矩阵直接计算PageRank"""
    n = adj_matrix.shape[0]
    out_degree = adj_matrix.sum(axis=0)

    # 列归一化 (简单实现)
    stochastic_matrix = np.copy(adj_matrix)
    for j in range(n):
        if out_degree[j] != 0:
            stochastic_matrix[:, j] /= out_degree[j]
        else:
            stochastic_matrix[:, j] = 1.0 / n  # 均匀分布

    teleport = (1 - damping) / n
    pr = np.full(n, 1.0 / n)

    for iteration in range(max_iter):
        pr_new = damping * stochastic_matrix @ pr + teleport

        delta = np.linalg.norm(pr_new - pr, 1)
        print(f"Iteration {iteration + 1}: delta = {delta:.6e}")

        if delta < tol:
            break
        pr = pr_new

    return pr


def save_result(pr_scores, node_list, output_path, top_k=100):
    """保存Top-K节点 PageRank结果"""
    ranked = sorted(zip(node_list, pr_scores), key=lambda x: x[1], reverse=True)
    with open(output_path, 'w') as f:
        for node_id, score in ranked[:top_k]:
            f.write(f"{node_id} {score:.8f}\n")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Naive PageRank Computation (Dense Matrix)')
    parser.add_argument('--input', type=str, default='Data.txt', help='Input edge list file')
    parser.add_argument('--output', type=str, default='Res1.txt', help='Output result file')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor (default 0.85)')
    args = parser.parse_args()

    print("Loading graph...")
    adj_matrix, node_list, node_id_map = load_graph(args.input)

    print(f"Graph loaded: {adj_matrix.shape[0]} nodes")

    print("Computing PageRank (Naive)...")
    pr_scores = pagerank(adj_matrix, damping=args.damping)

    print("Saving results...")
    save_result(pr_scores, node_list, args.output)

    print("Done! Top 100 results saved in", args.output)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")  # 输出总时间


if __name__ == "__main__":
    main()
