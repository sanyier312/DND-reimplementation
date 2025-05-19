import os
import numpy as np
from igraph import Graph
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def process_graph_file_paths(paths):
    input_path, output_path = paths
    g = Graph.Read_GraphML(input_path)
    g = compute_and_assign_node_features(g)
    g.write_graphml(output_path)
    print(f"✅ 已处理并保存: {output_path}")
    return output_path

def process_directory(input_dir, output_dir, max_workers=4):
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".gml"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            tasks.append((in_path, out_path))

    print(f"🚀 启动并行处理，共 {len(tasks)} 个图文件...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_graph_file_paths, t) for t in tasks]
        for future in as_completed(futures):
            future.result()  # Raise any exceptions

if __name__ == "__main__":
    input_dir = "./data/synthetic/test_large/ER"
    output_dir = "./data/synthetic/test_large/ER_with_features/"
    process_directory(input_dir, output_dir, max_workers=4)  # 根据CPU核心数适当调整
