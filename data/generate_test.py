import os
import random
from igraph import Graph
import numpy as np

# 全局配置
base_output_dir = "./data/synthetic/test_large/"
os.makedirs(base_output_dir, exist_ok=True)

node_sizes = [1000, 5000, 10000, 50000, 100000]
graphs_per_size = 10

# 通用保存函数
def save_graph(g, model, size, idx):
    model_dir = os.path.join(base_output_dir, model)
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{model}_n{size}_id{idx}.gml"
    g.write_graphml(os.path.join(model_dir, filename))


# === ER ===
def generate_er(n, p):
    return Graph.Erdos_Renyi(n=n, p=p, directed=True)

# === SBM ===
def generate_sbm(n, num_blocks, edge_prob_range):
    block_sizes = [n // num_blocks] * num_blocks
    block_sizes[-1] += n % num_blocks  # 处理余数
    probs = np.array([
        [round(random.uniform(*edge_prob_range), 3) for _ in range(num_blocks)]
        for _ in range(num_blocks)
    ])
    return Graph.SBM(n=sum(block_sizes), pref_matrix=probs, block_sizes=block_sizes, directed=True)

# === SPL ===
def generate_spl(n, m, exp_in, exp_out):
    return Graph.Static_Power_Law(
        n=n,
        m=m,
        exponent_in=exp_in,
        exponent_out=exp_out,
        loops=False,
        multiple=False,
        finite_size_correction=True
    )

# === 主循环 ===
for n in node_sizes:
    for i in range(graphs_per_size):
        print(f"Generating for n={n}, id={i}...")

        # ER
        er_p = random.uniform(0.0005, 0.005)  # 控制稀疏度
        er = generate_er(n, er_p)
        save_graph(er, "ER", n, i)

        # SBM
        num_blocks = random.randint(3, 6)
        sbm = generate_sbm(n, num_blocks, edge_prob_range=(0.01, 0.05))
        save_graph(sbm, "SBM", n, i)

        # SPL
        m = random.randint(n * 2, n * 3)  # 控制边数量为节点数的2~3倍
        exp_in = round(random.uniform(2.0, 2.9), 2)
        exp_out = round(random.uniform(2.0, 2.9), 2)
        spl = generate_spl(n, m, exp_in, exp_out)
        save_graph(spl, "SPL", n, i)

print("所有测试图生成完毕，保存在 ./data/synthetic/test_large/")
