import os
import itertools
import numpy as np
from igraph import Graph
import igraph
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 设置字体为黑体解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']


def draw_subgraph_with_deleted_nodes(original_g: Graph, delete_nodes, title=""):
    # 保留子图
    to_keep = [i for i in range(original_g.vcount()) if i not in delete_nodes]
    g_sub = original_g.subgraph(to_keep)
    sccs = g_sub.connected_components(mode="STRONG")
    largest_scc = max(sccs, key=len) if sccs else []

    # 设置节点颜色
    colors = []
    for v in range(g_sub.vcount()):
        if v in largest_scc:
            colors.append("lightgreen")
        else:
            colors.append("lightblue")
    g_sub.vs["color"] = colors

    # 可选：设置标签为原始索引（因为 subgraph 会改变索引）
    original_labels = [to_keep[i] for i in range(g_sub.vcount())]
    g_sub.vs["label"] = list(map(str, original_labels))

    layout = g_sub.layout("fr")  # Force-directed layout

    # 绘图设置
    visual_style = {
        "vertex_size": 30,
        "vertex_label_size": 12,
        "layout": layout,
        "bbox": (500, 400),
        "margin": 30,
        "edge_arrow_size": 0.6,
    }

    # 保存图像
    igraph.plot(g_sub, f"subgraph_{title}.png", **visual_style)

    # 显示额外信息：被删节点
    if delete_nodes:
        print(f"Deleted nodes (not shown in graph): {delete_nodes}")




# 更高效的 LSCC 相对大小计算函数
def compute_relative_lscc(g: Graph, n: int) -> float:
    if g.vcount() <= 1:
        return 0.0
    sccs = g.connected_components(mode="STRONG")
    if not sccs:
        return 0.0
    largest = max(map(len, sccs))
    return 0.0 if largest <= 1 else largest / n

# 单个组合评估函数（用于并行处理）
def evaluate_combination(args):
    graph, comb, n, threshold = args
    to_keep = [i for i in range(graph.vcount()) if i not in comb]
    g_sub = graph.subgraph(to_keep)

    # 快速剪枝：如果子图大小已经小于阈值，直接判定可行
    if g_sub.vcount() < threshold * n:
        return set(comb)
    
    rel = compute_relative_lscc(g_sub, n)
    if rel <= threshold:
        return set(comb)
    return None

# （并行 + 优化）
def compute_node_labels_strict(graph: Graph, threshold=0.1):
    n = graph.vcount()
    all_indices = list(range(n))
    optimal_sets = []
    min_k = None

    for k in range(1, n + 1):
        print(f"🔍 正在评估 k = {k} 的组合...")
        combs = list(itertools.combinations(all_indices, k))
        args = [(graph, comb, n, threshold) for comb in combs]

        with Pool(processes=8) as pool:  # 你可以根据机器调整进程数
            results = pool.map(evaluate_combination, args)

        current_valid_sets = [r for r in results if r]
        if current_valid_sets:
            optimal_sets = current_valid_sets
            min_k = k
            break

    print(f"\n✅ 找到最优解数量: {len(optimal_sets)}, 最小移除节点数: {min_k}")
    print(f"🌟 最优解展示: {optimal_sets[:]}")

    # 节点频率统计
    node_counts = defaultdict(int)
    for opt_set in optimal_sets:
        for node in opt_set:
            node_counts[node] += 1

    # 标签计算
    labels = []
    total = len(optimal_sets)
    for i in range(n):
        if node_counts[i] == total:
            labels.append(1.0)
        elif node_counts[i] > 0:
            labels.append(round(node_counts[i] / total, 4))
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
