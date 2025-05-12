import os
import random
from igraph import Graph

# 参数配置
output_dir = "./data/synthetic/ER/"
os.makedirs(output_dir, exist_ok=True)

node_count = 25
edge_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # 7 种概率   保证数据的均衡
graphs_per_prob = 10

def generate_er_graph(n, p):
    # 生成有向 ER 图（可能包含多个弱连接组件）
    return Graph.Erdos_Renyi(n=n, p=p, directed=True)

def save_graph(g, filename):
    g.write_graphml(filename)

# 主循环
total = 0
for p in edge_probs:
    for i in range(graphs_per_prob):
        g = generate_er_graph(node_count, p)
        filename = f"ER_n{node_count}_p{p:.2f}_id{i}.gml"
        save_graph(g, os.path.join(output_dir, filename))
        total += 1

print(f"共生成 {total} 个 ER 图，保存在 {output_dir}")
