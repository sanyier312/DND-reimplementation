import os
import time
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from data.dataloader import GraphMLDataset
from model.gat_model import GATDisintegrationModel

def load_model(model_path, in_channels, device, **model_kwargs):
    model = GATDisintegrationModel(in_channels=in_channels, **model_kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def test_er_runtime(model, er_dir, device):
    results = defaultdict(list)  # {size: [times]}
    files = sorted([f for f in os.listdir(er_dir) if f.endswith(".graphml")])
    dataset = GraphMLDataset(er_dir)

    for file in files:
        size = int(file.split("_n")[1].split("_")[0])
        data = next(d for d in dataset if os.path.basename(d.graph_name) == file)

        x = data["x"].to(device)
        edge_index = data["edge_index"].to(device)

        start = time.time()
        _ = model(x, edge_index)
        duration = time.time() - start

        results[size].append(duration)
        print(f"[ER] {file} | Nodes: {x.shape[0]} | Time: {duration:.4f} s")

    return results

def plot_er_runtime(results):
    sizes = sorted(results.keys())
    avg_times = [sum(results[size]) / len(results[size]) for size in sizes]

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, avg_times, marker='o', color='tab:blue', label="ER")
    plt.xlabel("Graph Size (Number of Nodes)")
    plt.ylabel("Average Inference Time (s)")
    plt.title("ER - Disintegration Prediction Time")
    plt.grid(True)
    plt.tight_layout()
    filename = "runtime_ER.png"
    plt.savefig(filename)
    plt.show()
    print(f"✅ Saved: {filename}")

def main():
    model_path = "best_model_er.pt"
    er_dir = "./data/synthetic/test_large/ER_with_features/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load sample to get in_channels
    sample_dataset = GraphMLDataset(er_dir)
    in_channels = sample_dataset[0]["x"].shape[1]

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

    print("⏱️ Running ER inference timing...")
    results = test_er_runtime(model, er_dir, device)
    plot_er_runtime(results)

if __name__ == "__main__":
    main()
