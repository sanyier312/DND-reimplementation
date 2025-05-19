import os
import torch
from torch.utils.data import Dataset, DataLoader
import igraph as ig
import numpy as np

class GraphMLDataset(Dataset):
    def __init__(self, graphml_dir, mode="train"):
        assert mode in ["train", "test"], "mode must be 'train' or 'test'"
        self.graph_paths = [
            os.path.join(graphml_dir, f) for f in os.listdir(graphml_dir)
            if f.endswith(".gml")
        ]
        self.mode = mode

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        path = self.graph_paths[idx]
        graph = ig.Graph.Read_GraphML(path)

        # 节点特征
        features = np.array([
            [
                v["in_degree"],
                v["out_degree"],
                v["betweenness"],
                v["pagerank"]
            ]
            for v in graph.vs
        ], dtype=np.float32)

        edge_index = np.array(graph.get_edgelist(), dtype=np.int64).T  # shape [2, num_edges]

        # 标签仅在训练模式中提供
        if self.mode == "train":
            labels = np.array(graph.vs["label"], dtype=np.float32)
        else:
            labels = None

        return {
            "x": torch.tensor(features),              # shape [num_nodes, 4]
            "y": torch.tensor(labels) if labels is not None else None,
            "edge_index": torch.tensor(edge_index)    # shape [2, num_edges]
        }

# ✅ 测试函数
def test_dataloader(mode="train"):
    dataset = GraphMLDataset("./data/synthetic/ER_with_features/", mode=mode)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(f"\n图 {i}：")
        print("特征 x:", batch["x"].shape)
        print("边 edge_index:", batch["edge_index"].shape)
        if batch["y"] is not None:
            print("标签 y:", batch["y"].shape)
        else:
            print("标签 y: None")

if __name__ == "__main__":
    test_dataloader(mode="train")  # or mode="test"
