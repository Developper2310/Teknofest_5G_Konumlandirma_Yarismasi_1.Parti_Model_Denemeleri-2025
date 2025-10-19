import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf


model_dir = "dist_model"
data_path = "test_coordinates_with_distance.xlsx"


target_pcis = [30, 40, 59, 48, 68, 76, 3, 13, 23]


features = ["RSRP", "RSRQ", "SINR"]
target = "Distance_m"


df = pd.read_excel(data_path)


for pci in target_pcis:
    model_path = os.path.join(model_dir, f"model_pci_{pci}.keras")

    if not os.path.exists(model_path):
        print(f" PCI {pci} için model bulunamadı, atlanıyor.")
        continue

    pci_df = df[df["PCI"] == pci]

    if len(pci_df) < 20:
        print(f" PCI {pci} için yeterli veri yok, atlanıyor.")
        continue

    X = pci_df[features].values
    y_true = pci_df[target].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = tf.keras.models.load_model(model_path)

    # Tahmin
    y_pred = model.predict(X_scaled).flatten()

    # Hata hesapla
    mae = mean_absolute_error(y_true, y_pred)
    print(f" PCI {pci} → Ortalama Hata (MAE): {mae/3:.2f} metre - Veri adedi: {len(pci_df)}")
