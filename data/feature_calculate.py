import os
import numpy as np
from igraph import Graph

def compute_and_assign_node_features(graph: Graph, damping=0.85):
    # 原属性自动保留，只添加新特征

    # 入度与出度
    graph.vs["in_degree"] = graph.indegree()
    graph.vs["out_degree"] = graph.outdegree()

    # 介数中心性（归一化）
    n = graph.vcount()
    if n > 2:
        betweenness = graph.betweenness(directed=True)
        betweenness = np.array(betweenness) / ((n - 1) * (n - 2))
    else:
        betweenness = [0.0] * n
    graph.vs["betweenness"] = betweenness

    # PageRank
    pageranks = graph.pagerank(directed=True, damping=damping)
    graph.vs["pagerank"] = pageranks

    return graph

def process_graph_file(input_path, output_path):
    g = Graph.Read_GraphML(input_path)
    g = compute_and_assign_node_features(g)
    g.write_graphml(output_path)
    print(f"✅ 已处理并保存: {output_path}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".gml"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            print(f"📂 处理图文件: {fname}")
            process_graph_file(in_path, out_path)

if __name__ == "__main__":
    input_dir = "./data/synthetic/ER_labeled/"
    output_dir = "./data/synthetic/ER_with_features/"
    process_directory(input_dir, output_dir)
