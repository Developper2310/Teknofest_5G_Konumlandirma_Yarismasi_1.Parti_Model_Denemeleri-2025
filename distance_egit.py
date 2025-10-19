import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


save_dir = "dist_model"
os.makedirs(save_dir, exist_ok=True)


train_path = "  _coordinates_with_distance.xlsx"


target_pcis = [30, 40, 59, 48, 68, 76, 3, 13, 23]


df = pd.read_excel(train_path)

features = ["RSRP", "RSRQ", "SINR"]
target = "Distance_m"

# Her hedef PCI için model eğit
for pci in target_pcis:
    pci_df = df[df["PCI"] == pci]

    if len(pci_df) < 20:
        print(f" PCI {pci} için yeterli veri yok, atlanıyor.")
        continue

    X = pci_df[features].values
    y = pci_df[target].values

    # Eğiti-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

   
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # model mimarisi
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),

        layers.Dense(32, activation='relu'),

        layers.Dense(1, activation='linear')  
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )

    print(f" PCI {pci} için mesafe tahmin modeli eğitiliyor...")

    history = model.fit(
        X_train_scaled, y_train,
        epochs=10000,       
        batch_size=32,
        validation_split=0.2,  
        verbose=1
    )

    model_path = os.path.join(save_dir, f"model_pci_{pci}.keras")
    model.save(model_path)
    print(f" PCI {pci} modeli '{model_path}' olarak kaydedildi.\n")
