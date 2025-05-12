import os
import random
from igraph import Graph

output_dir = "./data/synthetic/SPL/"
os.makedirs(output_dir, exist_ok=True)

node_count = 25
edge_options = [100, 150, 200]
exponent_range = (2.0, 2.9)
num_graphs = 150

def generate_spl_graph(n, m, exp_out, exp_in):
    return Graph.Static_Power_Law(
        n=n,
        m=m,
        exponent_out=exp_out,
        exponent_in=exp_in,
        loops=False,
        multiple=False,
        finite_size_correction=True,
    )

def save_graph(g, filename):
    g.write_graphml(filename)

for i in range(num_graphs):
    m = random.choice(edge_options)
    exp_out = round(random.uniform(*exponent_range), 2)
    exp_in = round(random.uniform(*exponent_range), 2)

    g = generate_spl_graph(node_count, m, exp_out, exp_in)

    filename = f"SPL_n{node_count}_m{m}_ein{exp_in}_eout{exp_out}_id{i}.gml"
    save_graph(g, os.path.join(output_dir, filename))

print(f"✅ 生成完毕：共 {num_graphs} 个 SPL 图，存储于 {output_dir}")
