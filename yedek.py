
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers




save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_excel("train_coordinates.xlsx")

features = ["RSRP", "RSRQ", "SINR", "Latitude", "Longitude"]

unique_pcis = df["PCI"].unique()

# Burada her PCI için ayrı model eğitilecek
for pci in unique_pcis:
    pci_df = df[df["PCI"] == pci]

    if len(pci_df) < 20:
        print(f" PCI {pci} için yeterli veri yok: {len(pci_df)} satır, atlanıyor.")
        continue

    X = pci_df[features].values
    y = pci_df[["Latitude", "Longitude"]].values

    # ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # mimari
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='linear')  # Latitude ve Longitude 
    ])

    # Modeli derle 
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    # Modeli eğit
    print(f" PCI {pci} için model eğitiliyor...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=1500,
        batch_size=32,
        validation_split=0.2,
        verbose=1 
    )

    #.keras uzantısı
    model_path = os.path.join(save_dir, f"model_pci_{pci}.keras")
    model.save(model_path)
    print(f" PCI {pci} için model {model_path} olarak kaydedildi.\n")
