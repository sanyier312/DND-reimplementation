import os
import itertools
import numpy as np
from igraph import Graph
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

# 设置字体为黑体解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']

def compute_relative_lscc(g: Graph, n) -> float:
    if g.vcount() == 0:
        return 0.0
    sccs = g.connected_components(mode="STRONG")
    if not sccs:
        return 0.0
    largest = max(len(comp) for comp in sccs)
    if largest == 1:
        return 0.0
    return largest / n

def draw_subgraph_with_deleted_nodes(original_g: Graph, delete_nodes, title=""):
    # 保留子图
    to_keep = [i for i in range(original_g.vcount()) if i not in delete_nodes]
    g_sub = original_g.subgraph(to_keep)
    sccs = g_sub.connected_components(mode="STRONG")
    largest_scc = max(sccs, key=len) if sccs else []

    # 转换为 NetworkX
    nx_g = nx.DiGraph()
    nx_g.add_edges_from(g_sub.get_edgelist())

    pos = nx.spring_layout(nx_g, seed=42)
    node_colors = []
    for v in nx_g.nodes():
        if v in largest_scc:
            node_colors.append("lightgreen")  # LSCC
        else:
            node_colors.append("lightblue")

    plt.figure(figsize=(5, 4))
    nx.draw_networkx_nodes(nx_g, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(nx_g, pos, arrows=True, arrowstyle='->')
    nx.draw_networkx_labels(nx_g, pos)

    # 画被删节点（灰红色虚节点）
    deleted_pos = {i: (0, 1.2 + 0.1 * idx) for idx, i in enumerate(delete_nodes)}
    nx.draw_networkx_nodes(nx_g, deleted_pos, nodelist=delete_nodes,
                           node_color='salmon', node_size=600, alpha=0.5)
    nx.draw_networkx_labels(nx_g, deleted_pos, labels={i: str(i) for i in delete_nodes})

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"subgraph_{title}.png", dpi=300)

def compute_node_labels_strict(graph: Graph, threshold=0.1):
    n = graph.vcount()
    all_indices = list(range(n))
    optimal_sets = []
    min_k = None

    for k in range(1, n + 1):
        current_valid_sets = []
        for comb in itertools.combinations(all_indices, k):
            to_keep = [i for i in range(n) if i not in comb]
            g_copy = graph.subgraph(to_keep)

            if compute_relative_lscc(g_copy, n) <= threshold:
                current_valid_sets.append(set(comb))
        if current_valid_sets:
            optimal_sets = current_valid_sets
            min_k = k
            break

    # print(f"\n✅ 找到最优解数量: {len(optimal_sets)}, 最小移除节点数: {min_k}")
    # print(f"🌟 最优解展示: {optimal_sets[:]}")

    # for idx, opt in enumerate(optimal_sets[:]):
    #     draw_subgraph_with_deleted_nodes(graph, delete_nodes=list(opt), title=f"optim#{idx+1}_delete_{opt}")

    node_counts = defaultdict(int)
    for opt_set in optimal_sets:
        for node in opt_set:
            node_counts[node] += 1

    labels = []
    for i in range(n):
        if node_counts[i] == len(optimal_sets):
            labels.append(1.0)
        elif node_counts[i] > 0:
            labels.append(round(node_counts[i] / len(optimal_sets), 4))
        else:
            labels.append(0.0)

    return labels

# ✅ 单元测试
def test_label_generator_on_simple_graph():
    print("\n🔍 正在运行标签生成单元测试...")
    g = Graph(directed=True)
    g.add_vertices(6)
    g.add_edges([
        (0, 1), (1, 2), (2, 0),  # SCC1
        (3, 4), (4, 5), (5, 3),  # SCC2
        (2, 3)  # 桥接边
    ])
    
    labels = compute_node_labels_strict(g, threshold=0.1)
    print("📋 生成的标签:", labels)

    expected = [0.3333] * 6  # 共9个最优解，每个节点出现3次
    for l, e in zip(labels, expected):
        assert abs(l - e) < 1e-3, f"标签不匹配: got {l}, expected {e}"

    print("✅ 测试通过：标签生成逻辑正确！")
def label_graph_file(input_path, output_path, threshold=0.1):
    g = Graph.Read_GraphML(input_path)
    labels = compute_node_labels_strict(g, threshold)
    g.vs["label"] = labels
    g.write_graphml(output_path)
def label_directory(input_dir, output_dir, threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith(".gml"):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        print(f"✔ 处理: {fname}")
        label_graph_file(in_path, out_path, threshold)

if __name__ == "__main__":
    # test_label_generator_on_simple_graph()
    
    input_dir = "./data/synthetic/ER/"
    output_dir = "./data/synthetic/ER_labeled/"
    label_directory(input_dir, output_dir, threshold=0.1)
