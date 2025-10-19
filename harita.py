import pandas as pd
import matplotlib.pyplot as plt

file_path = "SCANNER_ok.xlsx"  
df = pd.read_excel(file_path)

latitude_column = 'Latitude'   #  Latitude
longitude_column = 'Longitude' # Longitude

df = df.dropna(subset=[latitude_column, longitude_column])

# dağılım grafiği
plt.figure(figsize=(8, 6))
plt.scatter(df[longitude_column], df[latitude_column], s=5, alpha=0.5, c='blue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Konum Dağılımı')
plt.grid(True)
plt.show()