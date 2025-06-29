def read_result_file(filename):
    scores = {}
    with open(filename, 'r') as f:
        for line in f:
            node_id, score = line.strip().split()
            scores[int(node_id)] = float(score)
    return scores

# 读取结果
your_scores = read_result_file('Res.txt')
nx_scores = read_result_file('Res_nx.txt')

# Top-100 Precision
your_top100 = sorted(your_scores.items(), key=lambda x: -x[1])[:100]
your_top100_ids = {node_id for node_id, _ in your_top100}

nx_top100 = sorted(nx_scores.items(), key=lambda x: -x[1])[:100]
nx_top100_ids = {node_id for node_id, _ in nx_top100}

intersection = your_top100_ids & nx_top100_ids
precision_at_100 = len(intersection) / 100
print(f"Top-100 Precision: {precision_at_100:.4f}")

# L1误差
common_nodes = set(your_scores.keys()) & set(nx_scores.keys())
l1_error = sum(abs(your_scores[node] - nx_scores[node]) for node in common_nodes)
l1_avg = l1_error / len(common_nodes)

print(f"L1 error (sum absolute difference): {l1_error:.6f}")
print(f"Average absolute error (L1 / n): {l1_avg:.6e}")
