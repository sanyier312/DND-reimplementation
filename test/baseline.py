import os
import time
import csv
import numpy as np
from igraph import Graph
from tqdm import tqdm

def compute_relative_lscc(g: Graph, n: int) -> float:
    if g.vcount() <= 1:
        return 0.0
    sccs = g.connected_components(mode="STRONG")
    largest = max(map(len, sccs)) if sccs else 0
    return 0.0 if largest <= 1 else largest / n

def collapse_by_strategy(g: Graph, strategy: str, threshold=0.1, step_ratio=0.01, verbose=True):
    g = g.copy()
    node_count = g.vcount()
    removed_total = 0
    start = time.time()

    deletion_trace = []  # (total_removed_nodes, lscc_ratio)

    if verbose:
        print(f"\n Strategy: {strategy} | Original Nodes: {node_count}")

    while g.vcount() > 0:
        vcount = g.vcount()
        step_size = max(1, int(vcount * step_ratio))

        if strategy == "R-ID":
            scores = g.degree(mode="IN")
        elif strategy == "R-OD":
            scores = g.degree(mode="OUT")
        elif strategy == "R-SD":
            scores = [g.degree(i, mode="IN") + g.degree(i, mode="OUT") for i in range(vcount)]
        elif strategy == "PageRank":
            scores = g.pagerank(directed=True)
        elif strategy == "CoreHD":
            scores = [g.degree(i, mode="ALL") for i in range(vcount)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        sorted_nodes = np.argsort(-np.array(scores))[:step_size]
        g.delete_vertices(sorted_nodes)
        removed_total += len(sorted_nodes)

        lscc_ratio = compute_relative_lscc(g, node_count)
        deletion_trace.append((removed_total, round(lscc_ratio, 6)))

        if verbose:
            print(f"  ➤ Removed {removed_total} nodes so far | LSCC = {lscc_ratio:.4f}")

        if lscc_ratio <= threshold:
            duration = time.time() - start
            if verbose:
                print(f"  ✅ Disintegration complete after {removed_total} removals in {duration:.4f}s")
            return removed_total, duration, deletion_trace

    return removed_total, time.time() - start, deletion_trace

def save_deletion_trace(trace, output_dir, strategy, graph_name):
    trace_file = os.path.join(output_dir, f"deletion_trace_{strategy}_{graph_name}.csv")
    with open(trace_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RemovedNodes", "LSCC_Ratio"])
        writer.writerows(trace)

def run_baseline_on_dir(graph_dir, strategies, threshold=0.1, step_ratio=0.01, output_dir="baseline_results"):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".gml")])

    for strategy in strategies:
        csv_path = os.path.join(output_dir, f"results_{strategy}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Graph", "Size", "RemovedNodes", "TimeSeconds"])

            for fname in tqdm(files, desc=f"Processing {strategy}", ncols=80):
                size = int(fname.split("_n")[1].split("_")[0])
                if size >= 50000:
                    continue
                path = os.path.join(graph_dir, fname)
                g = Graph.Read_GraphML(path)

                print(f"\n📂 Processing: {fname} (size={size})")
                removed, duration, trace = collapse_by_strategy(g, strategy, threshold, step_ratio, verbose=True)

                writer.writerow([fname, size, removed, round(duration, 4)])

                graph_name = os.path.splitext(fname)[0]
                save_deletion_trace(trace, output_dir, strategy, graph_name)

    print(f"\n📁 All results saved to: {output_dir}")

if __name__ == "__main__":
    graph_dir = "./data/synthetic/test_large/ER_with_features/"
    strategies = ["R-ID", "R-OD", "R-SD", "PageRank", "CoreHD"]
    run_baseline_on_dir(graph_dir, strategies, step_ratio=0.01)
