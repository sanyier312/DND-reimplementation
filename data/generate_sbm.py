import os
import random
import numpy as np
from igraph import Graph

output_dir = "./data/synthetic/SBM/"
os.makedirs(output_dir, exist_ok=True)

node_count = 25
num_graphs = 80
block_range = [2, 3, 4, 5]
edge_prob_range = (0.1, 0.4)

def generate_sbm_graph(block_sizes, edge_prob_matrix):
    # 注意这里：igraph 要求 edge_prob_matrix 是一个 NumPy 矩阵
    return Graph.SBM(sum(block_sizes), edge_prob_matrix, block_sizes, directed=True)

def save_graph(g, filename):
    g.write_graphml(filename)

for i in range(num_graphs):
    num_blocks = random.choice(block_range)

    block_sizes = [1] * num_blocks
    for _ in range(node_count - num_blocks):
        block_sizes[random.randint(0, num_blocks - 1)] += 1

    probs = np.array([
        [round(random.uniform(*edge_prob_range), 3) for _ in range(num_blocks)]
        for _ in range(num_blocks)
    ])

    g = generate_sbm_graph(block_sizes, probs)

    filename = f"SBM_n{node_count}_b{num_blocks}_id{i}.gml"
    save_graph(g, os.path.join(output_dir, filename))

print(f"生成完毕：共 {num_graphs} 个 SBM 图，存储于 {output_dir}")
