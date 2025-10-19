import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np





class MLPModel(nn.Module):
    def __init__(self):  
        super(MLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# PCI listesi
target_pcis = [30, 40, 59, 48, 68, 76, 3, 13, 23]

test_df = pd.read_excel("test_angle_normalized.xlsx")

weight_dir = "mlp_weights"

results = []

for pci in target_pcis:
    pci_df = test_df[test_df["PCI"] == pci]

    if pci_df.empty:
        print(f" PCI {pci} için test verisi bulunamadı, atlanıyor.")
        continue


    X = pci_df[["RSRP", "RSRQ", "SINR"]].values.astype(np.float32)
    y_true = pci_df["angle"].values.astype(np.float32)

    model = MLPModel()
    weight_path = os.path.join(weight_dir, f"mlp_pci_{pci}.pth")
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # Tahmin
    with torch.no_grad():
        y_pred = model(torch.tensor(X)).squeeze().numpy()

    # MAE, RMSE
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    results.append({
        "PCI": pci,
        "Test Sample Count": len(y_true),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })



results_df = pd.DataFrame(results)
results_df.to_excel("mlp_test_results.xlsx", index=False)

print(" Tüm modeller test edildi. Sonuçlar: 'mlp_test_results.xlsx'")