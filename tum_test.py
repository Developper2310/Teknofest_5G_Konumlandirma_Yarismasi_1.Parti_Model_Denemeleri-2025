import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from geopy.distance import geodesic




df = pd.read_excel("test_angle_normalized.xlsx")
features = ["RSRP", "RSRQ", "SINR", "BS_Latitude", "BS_Longitude"]
unique_pcis = df["PCI"].unique()

results = []

for pci in unique_pcis:
    pci_df = df[df["PCI"] == pci]

    if len(pci_df) < 20:
        print(f" PCI {pci} için yeterli veri yok, atlanıyor.")
        continue

    X = pci_df[features].values
    y = pci_df[["User_Latitude", "User_Longitude"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_path = f"models/model_pci_{pci}.keras"
    if not os.path.exists(model_path):
        print(f"Model dosyası bulunamadı: {model_path}")
        continue

    model = load_model(model_path)
    y_pred = model.predict(X_test_scaled, verbose=0)

    for i in range(len(y_test)):
        true_loc = (y_test[i][0], y_test[i][1])
        pred_loc = (y_pred[i][0], y_pred[i][1])
        distance_m = geodesic(true_loc, pred_loc).meters

        results.append({
            "PCI": pci,
            "True_Latitude": true_loc[0],
            "True_Longitude": true_loc[1],
            "Pred_Latitude": pred_loc[0],
            "Pred_Longitude": pred_loc[1],
            "Distance_Error_m": round(distance_m, 2)
        })


results_df = pd.DataFrame(results)
results_df.to_excel("location_prediction_results.xlsx", index=False)

# Ortalama hata
mean_error = results_df["Distance_Error_m"].mean()
print(f"\n Ortalama konum hatası: {mean_error:.2f} metre")
print("Tahmin sonuçları 'location_prediction_results.xlsx' dosyasına kaydedildi.")
