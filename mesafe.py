import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2





def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Dünya yarıçapı 
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def rsrp_to_distance(rsrp_dbm, freq_mhz=3500):

    tx_power_dbm = 16.91  # 20W = 43 dBm
    path_loss_db = tx_power_dbm - rsrp_dbm
    freq_hz = freq_mhz * 1e6
    c = 3e8  # ışık hızı
    wavelength = c / freq_hz

    distance = wavelength / (4 * np.pi) * 10**(path_loss_db / 20)
    return distance


scanner_df = pd.read_excel("SCANNER_ok.xlsx")
cells_df = pd.read_excel("İTÜ 5G Hücre Bilgileri.xlsx")
cells_df.columns = cells_df.columns.str.strip() 

target_pcis = [30, 40, 59, 48, 68, 76, 3, 13, 23]


results = []

for idx, row in scanner_df.iterrows():
    try:
        pci = row["NR_Scan_PCI_SortedBy_RSRP_0"]
        if pci not in target_pcis:
            continue

        rsrp = row["NR_Scan_SSB_RSRP_SortedBy_RSRP_0"]
        user_lat = row["Latitude"]
        user_lon = row["Longitude"]

        cell = cells_df[cells_df["PCI"] == pci]
        if cell.empty:
            continue

        bs_lat = cell.iloc[0]["Latitude"]
        bs_lon = cell.iloc[0]["Longitude"]

        real_distance = haversine(user_lat, user_lon, bs_lat, bs_lon)
        predicted_distance = rsrp_to_distance(rsrp)

        results.append({
            "PCI": pci,
            "User_Latitude": user_lat,
            "User_Longitude": user_lon,
            "Tahmini_Mesafe_m": round(predicted_distance, 2),
            "Gerçek_Mesafe_m": round(real_distance, 2)
        })

    except Exception as e:
        print(f"Hata (satır {idx}): {e}")
        continue


df_result = pd.DataFrame(results)
df_result.to_excel("mesafe_karsilastirma.xlsx", index=False)
print(" 'mesafe_karsilastirma.xlsx' dosyası oluşturuldu.")