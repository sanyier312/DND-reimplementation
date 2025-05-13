import torch
import torch.nn.functional as F
from torch.nn import Linear, ELU, Sigmoid, ModuleList, Dropout
from torch_geometric.nn import GATConv
from data.dataloader import GraphMLDataset

class GATDisintegrationModel(torch.nn.Module):
    def __init__(self, in_channels, gat_hidden_channels, gat_heads, num_gat_layers,
                 mlp_hidden_channels, mlp_layers, dropout=0.2):
        super().__init__()
        self.gat_layers = ModuleList()
        self.residuals = ModuleList()
        self.elu = ELU()
        self.dropout = Dropout(dropout)

        # 构建 GAT 层
        for i in range(num_gat_layers):
            input_dim = in_channels if i == 0 else gat_hidden_channels * gat_heads
            conv = GATConv(input_dim, gat_hidden_channels, heads=gat_heads, dropout=dropout)
            self.gat_layers.append(conv)
            self.residuals.append(Linear(input_dim, gat_hidden_channels * gat_heads))

        # 构建 MLP
        self.mlp = ModuleList()
        input_dim = gat_hidden_channels * gat_heads
        for i in range(mlp_layers):
            out_dim = mlp_hidden_channels if i < mlp_layers - 1 else 1
            self.mlp.append(Linear(input_dim, out_dim))
            input_dim = out_dim

        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index):
        # GAT 层 + 残差 + ELU
        for gat, res in zip(self.gat_layers, self.residuals):
            h = gat(x, edge_index)
            x = self.elu(h + res(x))
            x = self.dropout(x)

        # MLP 层
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:
                x = self.elu(x)

        # 输出归一化到 [0, 1]
        return self.sigmoid(x).squeeze(-1)

if __name__ == "__main__":
    dataset = GraphMLDataset("./data/synthetic/ER_with_features/")
    sample = dataset[0]  # 取第一个图

    x = sample["x"]
    edge_index = sample["edge_index"]
    y = sample["y"]

    print("输入特征维度:", x.shape)
    print("边索引维度:", edge_index.shape)
    print("标签维度:", y.shape)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GATDisintegrationModel(
        in_channels=x.shape[1],
        gat_hidden_channels=8,
        gat_heads=4,
        num_gat_layers=2,
        mlp_hidden_channels=16,
        mlp_layers=2,
        dropout=0.2
    ).to(device)

    # 数据转移到设备
    x = x.to(device)
    edge_index = edge_index.to(device)

    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)

    print("✅ 模型输出维度:", output.shape)
    print("部分输出值:", output[:])


    
