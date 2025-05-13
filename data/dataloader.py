import os
import torch
from torch.utils.data import Dataset, DataLoader
import igraph as ig
import numpy as np

class GraphMLDataset(Dataset):
    def __init__(self, graphml_dir):
        self.graph_paths = [
            os.path.join(graphml_dir, f) for f in os.listdir(graphml_dir)
            if f.endswith(".gml")
        ]

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        path = self.graph_paths[idx]
        graph = ig.Graph.Read_GraphML(path)

        # 节点特征顺序固定：in_degree, out_degree, betweenness, pagerank
        features = np.array([
            [
                v["in_degree"],
                v["out_degree"],
                v["betweenness"],
                v["pagerank"]
            ]
            for v in graph.vs
        ], dtype=np.float32)

        labels = np.array(graph.vs["label"], dtype=np.float32)
        edge_index = np.array(graph.get_edgelist(), dtype=np.int64).T  # shape [2, num_edges]

        return {
            "x": torch.tensor(features),              # shape [num_nodes, 4]
            "y": torch.tensor(labels),                # shape [num_nodes]
            "edge_index": torch.tensor(edge_index)    # shape [2, num_edges]
        }
def test_dataloader():
    dataset = GraphMLDataset("./data/synthetic/ER_with_features/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(f"\n图 {i}：")
        print("特征 x:", batch["x"].shape)
        print("特征 x:", batch["x"])
        print("标签 y:", batch["y"].shape)
        print("标签 y:", batch["y"])
        print("边 edge_index:", batch["edge_index"].shape)
        if i == 1: break  # 测试前两个图即可
if __name__ == "__main__":
    test_dataloader()