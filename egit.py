# train_mlp_models.py
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split





class MLP(nn.Module):
    def __init__(self):  
        super(MLP, self).__init__()  
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)



df = pd.read_excel("train_angle_normalized.xlsx")


unique_pcis = df["PCI"].unique()


os.makedirs("mlp_weights", exist_ok=True)

# Her PCI için ayrı model eğitilecek
for pci in unique_pcis:
    pci_df = df[df["PCI"] == pci]
    
    X = pci_df[["RSRP", "RSRQ", "SINR"]].values.astype(np.float32)
    y = pci_df["angle"].values.astype(np.float32).reshape(-1, 1)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = MLP()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Parametreler
    for epoch in range(5000):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    torch.save(model.state_dict(), f"mlp_weights/mlp_pci_{pci}.pth")
    print(f" PCI {pci} için model eğitildi ve kaydedildi.")