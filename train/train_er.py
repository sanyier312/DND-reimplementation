import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataloader import GraphMLDataset
from model.gat_model import GATDisintegrationModel  
import wandb
import os

def train_model(
    data_path,
    num_epochs=50,
    lr=1e-3,
    weight_decay=5e-4,
    gat_hidden_channels=8,
    gat_heads=4,
    num_gat_layers=2,
    mlp_hidden_channels=16,
    mlp_layers=2,
    dropout=0.2,
    model_save_path="best_model.pt"
):
    # 初始化 wandb
    wandb.init(project="graph-model-training", config={
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "gat_hidden_channels": gat_hidden_channels,
        "gat_heads": gat_heads,
        "num_gat_layers": num_gat_layers,
        "mlp_hidden_channels": mlp_hidden_channels,
        "mlp_layers": mlp_layers,
        "dropout": dropout,
    })

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataset = GraphMLDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # 初始化模型
    sample = dataset[0]
    in_channels = sample["x"].shape[1]
    model = GATDisintegrationModel(
        in_channels=in_channels,
        gat_hidden_channels=gat_hidden_channels,
        gat_heads=gat_heads,
        num_gat_layers=num_gat_layers,
        mlp_hidden_channels=mlp_hidden_channels,
        mlp_layers=mlp_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            x = batch["x"].squeeze(0).to(device)
            edge_index = batch["edge_index"].squeeze(0).to(device)
            y = batch["y"].squeeze(0).to(device)

            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

        # 使用 wandb 记录损失值
        wandb.log({"epoch": epoch, "loss": avg_loss})

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"🔺 Saved best model (loss={best_loss:.6f})")

    print("✅ 训练完成")
    wandb.finish()

if __name__ == "__main__":
    train_model("./data/synthetic/SBM_with_features/")
