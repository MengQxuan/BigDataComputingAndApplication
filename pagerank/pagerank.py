import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import argparse
import time

def load_graph(file_path):
    """加载边列表并构建邻接矩阵"""
    edges = []
    node_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            from_node, to_node = map(int, line.strip().split())
            edges.append((from_node, to_node))
            node_set.update([from_node, to_node])

    node_list = sorted(list(node_set))
    node_id_map = {node: idx for idx, node in enumerate(node_list)}

    row = []
    col = []
    for from_node, to_node in edges:
        row.append(node_id_map[to_node])   # 注意PageRank是基于反向边构图 (M.T)
        col.append(node_id_map[from_node])

    data = np.ones(len(row))
    n = len(node_list)
    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)
    
    return adj_matrix.tocsr(), node_list, node_id_map


def pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """优化CSR稀疏矩阵和向量化PageRank计算"""
    n = adj_matrix.shape[0]
    out_degree = np.array(adj_matrix.sum(axis=0)).flatten()

    # 处理孤立节点 (出度为0)
    dangling_nodes = (out_degree == 0)

    # 列归一化 (出边归一化)
    col_sum = out_degree
    col_sum[col_sum == 0] = 1  # 避免除0
    inv_out_degree = sp.diags(1.0 / col_sum)
    stochastic_matrix = adj_matrix @ inv_out_degree

    teleport = (1 - damping) / n
    pr = np.full(n, 1.0 / n)

    # 计算PageRank
    for iteration in range(max_iter):
        # 矩阵-向量乘法，向量化计算
        pr_new = damping * stochastic_matrix.dot(pr) + teleport * np.ones_like(pr)

        # 处理 dangling nodes
        dangling_weight = pr[dangling_nodes].sum() / n
        pr_new += damping * dangling_weight

        # 检查收敛
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
    parser = argparse.ArgumentParser(description='PageRank Computation with CSR Sparse Matrix + Vectorization')
    parser.add_argument('--input', type=str, default='Data.txt', help='Input edge list file')
    parser.add_argument('--output', type=str, default='Res.txt', help='Output result file')
    parser.add_argument('--damping', type=float, default=0.85, help='Damping factor (default 0.85)')
    args = parser.parse_args()

    print("Loading graph...")
    adj_matrix, node_list, node_id_map = load_graph(args.input)

    print(f"Graph loaded: {adj_matrix.shape[0]} nodes, {adj_matrix.nnz} edges")
    
    print("Computing PageRank...")
    pr_scores = pagerank(adj_matrix, damping=args.damping)

    print("Saving results...")
    save_result(pr_scores, node_list, args.output)

    print("Done! Top 100 results saved in", args.output)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")  # 输出总时间


if __name__ == "__main__":
    main()