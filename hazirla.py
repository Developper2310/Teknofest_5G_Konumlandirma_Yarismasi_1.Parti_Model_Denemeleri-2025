import pandas as pd
import numpy as np
from math import atan2, degrees
from sklearn.model_selection import train_test_split


scanner_df = pd.read_excel("SCANNER_ok.xlsx")
cells_df = pd.read_excel("İTÜ 5G Hücre Bilgileri.xlsx")


cells_df.columns = cells_df.columns.str.strip()


target_pcis = [30, 40, 59, 48, 68, 76, 3, 13, 23]


dropped_logs = []
all_data = []


for idx, row in scanner_df.iterrows():
    for i in range(5): 
        try:
            pci = row[f"NR_Scan_PCI_SortedBy_RSRP_{i}"]
            rsrp = row[f"NR_Scan_SSB_RSRP_SortedBy_RSRP_{i}"]
            rsrq = row[f"NR_Scan_SSB_RSRQ_SortedBy_RSRP_{i}"]
            sinr = row[f"NR_Scan_SSB_SINR_SortedBy_RSRP_{i}"]
            user_lon = row["Longitude"]
            user_lat = row["Latitude"]
        except KeyError:
            dropped_logs.append({
                "index": idx,
                "pci_index": i,
                "reason": f"Sütun eksik"
            })
            continue


        if pd.isnull([pci, rsrp, rsrq, sinr, user_lon, user_lat]).any():
            dropped_logs.append({
                "index": idx,
                "pci_index": i,
                "reason": f"Eksik veri"
            })
            continue

        if pci not in target_pcis:
            dropped_logs.append({
                "index": idx,
                "pci_index": i,
                "reason": f"PCI ({pci}) hedef listede değil"
            })
            continue

        cell_info = cells_df[cells_df["PCI"] == pci]
        if cell_info.empty:
            dropped_logs.append({
                "index": idx,
                "pci_index": i,
                "reason": f"PCI ({pci}) için hücre bilgisi yok"
            })
            continue

        bs_lat = cell_info.iloc[0]["Latitude"]
        bs_lon = cell_info.iloc[0]["Longitude"]
        azimuth = cell_info.iloc[0]["Azimuth [°]"]

        # Açı hesaplama
        dx = user_lon - bs_lon
        dy = user_lat - bs_lat
        direction_angle = degrees(atan2(dy, dx)) % 360

        # Görüşün merkez ışını
        zero_ray = (azimuth - 32.5 + 360) % 360

        # (0–360)
        relative_angle = (direction_angle - zero_ray + 360) % 360

        if relative_angle > 65:
            dropped_logs.append({
                "index": idx,
                "pci_index": i,
                "reason": f"Görüş alanı dışında: {relative_angle:.2f}°"
            })
            continue

        all_data.append({
            "PCI": pci,
            "RSRP": rsrp,
            "RSRQ": rsrq,
            "SINR": sinr,
            "angle": relative_angle,
            "User_Longitude": user_lon,
            "User_Latitude": user_lat,
            "BS_Longitude": bs_lon,
            "BS_Latitude": bs_lat
        })


final_df = pd.DataFrame(all_data)

# Eğitim-test 
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

train_df.to_excel("train_angle_normalized.xlsx", index=False)
test_df.to_excel("test_angle_normalized.xlsx", index=False)
pd.DataFrame(dropped_logs).to_excel("dropped_rows_log.xlsx", index=False)

print(f"✅ Toplam {len(final_df)} satır işlendi ve veri setlerine ayrıldı.")
print(f"⚠ {len(dropped_logs)} satır atlandı. Detaylar: 'dropped_rows_log.xlsx'")