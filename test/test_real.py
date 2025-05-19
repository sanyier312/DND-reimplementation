import os
import time
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
from data.dataloader import GraphMLDataset
from model.gat_model import GATDisintegrationModel
from igraph import Graph

def load_model(model_path, in_channels, device, **model_kwargs):
    model = GATDisintegrationModel(in_channels=in_channels, **model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def compute_relative_lscc(g: Graph, n: int) -> float:
    if g.vcount() <= 1:
        return 0.0
    sccs = g.connected_components(mode="STRONG")
    largest = max(map(len, sccs)) if sccs else 0
    return 0.0 if largest <= 1 else largest / n

@torch.no_grad()
def test_specific_graph_disintegration(model, dataset, target_filename, device,
                                       threshold=0.1, step_ratio=0.01,
                                       output_dir="model_results"):
    os.makedirs(output_dir, exist_ok=True)

    # 找到指定图的索引
    matched_index = None
    for i, path in enumerate(dataset.graph_paths):
        if os.path.basename(path) == target_filename:
            matched_index = i
            break
    if matched_index is None:
        raise FileNotFoundError(f"指定图 {target_filename} 不在 dataset.graph_paths 中")

    graph_path = dataset.graph_paths[matched_index]
    graph_name = os.path.splitext(target_filename)[0]
    results_file = os.path.join(output_dir, f"model_{graph_name}_results.csv")

    g = Graph.Read_GraphML(graph_path)
    size = g.vcount()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(dataloader):
        if i != matched_index:
            continue

        x = batch["x"].squeeze(0).to(device)
        edge_index = batch["edge_index"].squeeze(0).to(device)

        print(f"\n Processing {target_filename} (size={size})...")
        start = time.time()

        pred = model(x, edge_index).cpu().numpy()
        sorted_nodes = np.argsort(-pred.flatten())
        step_size = max(1, int(size * step_ratio))

        trace = []

        for k in range(step_size, size + 1, step_size):
            to_remove = sorted_nodes[:k].tolist()
            g_tmp = g.copy()
            g_tmp.delete_vertices(to_remove)

            lscc_ratio = compute_relative_lscc(g_tmp, size)
            trace.append((k, round(lscc_ratio, 6)))

            print(f"  ➤ Removed {k} nodes | LSCC ratio = {lscc_ratio:.4f}")

            if lscc_ratio <= threshold:
                duration = time.time() - start
                with open(results_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Graph", "Size", "RemovedNodes", "TimeSeconds"])
                    writer.writerow([target_filename, size, k, round(duration, 4)])

                print(f"  ✅ Disintegration achieved after removing {k} nodes")
                print(f"  ⏱️  Time used: {duration:.4f} seconds")

                save_trace(trace, output_dir, f"model_{graph_name}")
                break

def save_trace(trace, output_dir, trace_name):
    trace_path = os.path.join(output_dir, f"{trace_name}_trace.csv")
    with open(trace_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RemovedNodes", "LSCC_Ratio"])
        writer.writerows(trace)


def main():
    model_path = "best_model_spl.pt"
    graph_dir = "./data/real/"  # 包含多个图的目录
    target_filename = "wiki_vote.gml"  # 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GraphMLDataset(graph_dir, mode="test")

    # 获取特征维度
    index = [i for i, p in enumerate(dataset.graph_paths)
             if os.path.basename(p) == target_filename]
    if not index:
        raise FileNotFoundError(f"{target_filename} 不在 {graph_dir} 中")
    in_channels = dataset[index[0]]["x"].shape[1]

    model = load_model(
        model_path=model_path,
        in_channels=in_channels,
        device=device,
        gat_hidden_channels=8,
        gat_heads=4,
        num_gat_layers=2,
        mlp_hidden_channels=16,
        mlp_layers=2,
        dropout=0.2
    )

    test_specific_graph_disintegration(model, dataset, target_filename, device)

    graph_name = os.path.splitext(target_filename)[0]
    results_file = f"model_results/model_{graph_name}_results.csv"
    

if __name__ == "__main__":
    main()
