import numpy as np
import argparse
import time

def load_graph(file_path):
    """仅加载边列表和节点集合"""
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

    return edges, node_list, node_id_map, n

def compute_out_degree(edges, node_id_map, n):
    """计算每个节点出度"""
    out_degree = np.zeros(n, dtype=np.float64)
    for from_node, _ in edges:
        j = node_id_map[from_node]
        out_degree[j] += 1
    return out_degree

def pagerank_blockwise(edges, node_id_map, n, out_degree, damping=0.85, max_iter=100, tol=1e-6, block_size=512):
    """边列表 + 分块生成矩阵块 → Block Matrix-Vector 乘法 PageRank"""
    teleport = (1 - damping) / n
    pr = np.full(n, 1.0 / n)

    for iteration in range(max_iter):
        pr_new = np.zeros(n, dtype=np.float64)

        # 外层block (行block)
        for row_start in range(0, n, block_size):
            row_end = min(row_start + block_size, n)
            partial_sum = np.zeros(row_end - row_start, dtype=np.float64)

            # 只处理 relevant edges 属于当前行块
            relevant_edges = [
                (from_node, to_node)
                for from_node, to_node in edges
                if row_start <= node_id_map[to_node] < row_end
            ]

            # 模拟“块矩阵乘法”
            for from_node, to_node in relevant_edges:
                i = node_id_map[to_node] - row_start
                j = node_id_map[from_node]

                # stochastic_matrix[i,j] = adj[i,j] / out_degree[j]
                if out_degree[j] != 0:
                    contrib = 1.0 / out_degree[j]
                else:
                    contrib = 1.0 / n  # 死链

                partial_sum[i] += contrib * pr[j]

            pr_new[row_start:row_end] = damping * partial_sum + teleport

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
    parser = argparse.ArgumentParser(description='PageRank (True Block Matrix Computation)')
    parser.add_argument('--input', type=str, default='Data.txt', help='Input edge list file')
    parser.add_argument('--output', type=str, default='Res2.txt', help='Output result file')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor (default 0.85)')
    parser.add_argument('--block_size', type=int, default=512, help='Block size (default 512)')
    args = parser.parse_args()

    print("Loading graph...")
    edges, node_list, node_id_map, n = load_graph(args.input)
    print(f"Graph loaded: {n} nodes, {len(edges)} edges")

    print("Computing out-degree...")
    out_degree = compute_out_degree(edges, node_id_map, n)

    print("Computing PageRank (Block-wise)...")
    pr_scores = pagerank_blockwise(edges, node_id_map, n, out_degree, damping=args.damping, block_size=args.block_size)

    print("Saving results...")
    save_result(pr_scores, node_list, args.output)

    print("Done! Top 100 results saved in", args.output)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
