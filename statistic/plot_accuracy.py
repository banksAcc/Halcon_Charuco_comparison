import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
from glob import glob

# Leggi il CSV
df_subpixel = pd.read_csv("accuracy_graph&data/accuracy_charuco_sub.csv")
df_normal = pd.read_csv("accuracy_graph&data/accuracy_charuco.csv")
df_halcon = pd.read_csv("accuracy_graph&data/accuracy_halcon.csv")

# Estrai colonna 11 (ERRORE DELLA RILEVAZIONE)
valori_subpixel = df_subpixel.iloc[:, 7].abs()
valori_normal = df_normal.iloc[:, 7].abs()
valori_halcon = df_halcon.iloc[:,7].abs()
# Calcola le medie
media_subpixel = valori_subpixel.mean()
media_normal = valori_normal.mean()
media_halcon = valori_halcon.mean()
# Calcola dev standard
devst_subpixel = valori_subpixel.std()
devst_normal = valori_normal.std()
devst_halcon = valori_halcon.std()
# X: numeri da 1 a N
x_sub = range(1, len(valori_subpixel) + 1)
x_norm = range(1, len(valori_normal) + 1)
x_halcon = range(1, len(valori_halcon) + 1)
# Estrai colonna 10 (RILEVAZIONE EFFETTUATA)
dist_mis_sub = df_subpixel.iloc[:, 5].abs()
dist_mis_norm = df_normal.iloc[:, 5].abs()
dist_mis_halcon = df_halcon.iloc[:,5].abs()
# Calcola le medie
media_dist_subpixel = dist_mis_sub.mean()
media_dist_normal = dist_mis_norm.mean()
media_dist_halcon = dist_mis_halcon.mean()
# X: numeri da 1 a N
x_dist_sub = range(1, len(dist_mis_sub) + 1)
x_dist_norm = range(1, len(dist_mis_norm) + 1)
x_dist_halcon = range(1, len(dist_mis_halcon) + 1)


# 1° GRAFICO
plt.figure(figsize=(20, 6))
plt.plot(x_norm,    valori_normal,    marker='o', linestyle='-', color='#d39039',
         label=f'ChArUco (μ={media_normal:.4f})', linewidth=1)
plt.plot(x_halcon,  valori_halcon,    marker='o', linestyle='-', color='#91d64d',
         label=f'Halcon (μ={media_halcon:.4f})', linewidth=1)
# linee medie
plt.axhline(media_normal, color='#d39039', linestyle='--', linewidth=1)
plt.axhline(media_halcon, color='#91d64d', linestyle='--', linewidth=1)

# linee verticali
plt.axvline(200, color='red',   linestyle='-.', linewidth=1, label='+10/0')
plt.axvline(400, color='blue',  linestyle='-.', linewidth=1, label='-10/0')
plt.axvline(600, color='green', linestyle='-.', linewidth=1, label='+10/-10')

plt.ylim(0, max(max(valori_normal), max(valori_halcon)) + 0.01)
plt.xlabel("Campioni")
plt.ylabel("Errore (mm)")
plt.title("Variazione dell'errore - Accuratezza")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()


# 2° GRAFICO
plt.figure(figsize=(20, 5))
plt.plot(x_sub,   valori_subpixel, marker='o', linestyle='-', color='#3976d3',
         label=f'ChArUco SubPixel (μ={media_subpixel:.4f})', linewidth=1)
plt.plot(x_norm,  valori_normal,    marker='o', linestyle='-', color='#d39039',
         label=f'ChArUco (μ={media_normal:.4f})',    linewidth=1)
# linee medie
plt.axhline(media_subpixel, color='#3976d3', linestyle='--', linewidth=1)
plt.axhline(media_normal,   color='#d39039', linestyle='--', linewidth=1)

# linee verticali
plt.axvline(200, color='red',   linestyle='-.', linewidth=1, label='+10/0')
plt.axvline(400, color='blue',  linestyle='-.', linewidth=1, label='-10/0')
plt.axvline(600, color='green', linestyle='-.', linewidth=1, label='+10/-10')

plt.xlabel("Campioni")
plt.ylabel("Errore (mm)")
plt.title("ChArUco vs ChArUco SubPixel - Accuratezza")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()


# --- TERZO GRAFICO: primi 400 campioni, VALORE MISURATO Halcon vs. ChArUco (normale)
x1 = np.arange(1, 401)
y1_norm = dist_mis_norm[:400]
y1_hal  = dist_mis_halcon[:400]
media_dist_normal_400 =  y1_norm.mean()
media_dist_halcon_400 =  y1_hal.mean()

plt.figure(figsize=(12, 5))
plt.plot(x1, y1_norm, marker='o', linestyle='-', color='#d39039', label=f'ChArUco (μ={media_dist_normal_400:.4f})')
plt.plot(x1, y1_hal,  marker='o', linestyle='-', color='#91d64d', label=f'Halcon (μ={media_dist_halcon_400:.4f})')

# linea orizzontale true = 100
plt.axhline(100, color='red', linestyle='--', linewidth=1, label='100 mm')

# medie
plt.axhline(media_dist_normal_400, color='#d39039', linestyle=':', linewidth=1)
plt.axhline(media_dist_halcon_400,  color='#91d64d', linestyle=':', linewidth=1)

# linee verticali alle transizioni
plt.axvline(200, color='grey', linestyle='-.', linewidth=1, label='+10/0')
plt.axvline(400, color='grey', linestyle='-.', linewidth=1, label='-10/0')

plt.xlabel("Campione")
plt.ylabel("Δ distanza (mm)")
plt.title("Valore misurato vs. reale - Accuratezza")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()


# --- QUARTO GRAFICO: successivi 200 campioni, VALORE MISURATO Halcon vs. ChArUco (normale)
x2 = np.arange(1, 201)
y2_norm = dist_mis_norm[400:600]
y2_hal  = dist_mis_halcon[400:600]
media_dist_normal_200 =  y2_norm.mean()
media_dist_halcon_200 =  y2_hal.mean()

plt.figure(figsize=(12, 5))
plt.plot(x2, y2_norm, marker='o', linestyle='-', color='#d39039', label=f'ChArUco (μ={media_dist_normal_200:.4f})')
plt.plot(x2, y2_hal,  marker='o', linestyle='-', color='#91d64d', label=f'Halcon (μ={media_dist_halcon_200:.4f})')

# linea orizzontale true = 200
plt.axhline(200, color='red', linestyle='--', linewidth=1, label='200 mm')

# medie
plt.axhline(media_dist_normal_200, color='#d39039', linestyle=':', linewidth=1)
plt.axhline(media_dist_halcon_200,  color='#91d64d', linestyle=':', linewidth=1)

plt.xlabel("Campione")
plt.ylabel("Δ distanza (mm)")
plt.title("Valore misurato vs. reale - Accuratezza")
plt.legend()
plt.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()

# QUINTO GRAFICO – Deviazione standard
# --- Dati ---
metodi  = ['ChArUco', 'ChArUco Sub', 'Halcon']
dev_std = [devst_normal, devst_subpixel, devst_halcon]

error_kw = dict(
    lw=1,      # spessore linee error bar
    capsize=5, # lunghezza alette
    ecolor='red'
)

labels = [
    f"ChArUco: {devst_normal:.3f}",
    f"ChArUco Sub: {devst_subpixel:.3f}",
    f"Halcon: {devst_halcon:.3f}"
]

# --- Crea una sola figura ---
plt.figure(figsize=(10, 6))

# --- Unico plotting delle barre ---
bars = plt.bar(
    metodi, dev_std,
    color=['#d39039', '#3976d3', '#91d64d'],
    error_kw=error_kw
)

# --- Legenda con i valori ---
plt.legend(bars, labels, title="Metodo (Dev. Std.)")

# --- Etichette e titolo ---
plt.ylabel("Variazione (mm)")
plt.title("Stima della stabilità per metodo")
plt.grid(True, axis='y', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.show()