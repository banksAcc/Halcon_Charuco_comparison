import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
from glob import glob

# Leggi il CSV
df_subpixel = pd.read_csv("../output/set_+10_charuco_sub.csv")
df_normal = pd.read_csv("../output/set_+10_charuco.csv")
df_halcon = pd.read_csv("../output/set_+10_halcon.csv")

# Estrai colonna 11 (ERRORE DELLA RILEVAZIONE)
valori_subpixel = df_subpixel.iloc[:, 11].abs()
valori_normal = df_normal.iloc[:, 11].abs()
valori_halcon = df_halcon.iloc[:,11].abs()
# Calcola le medie
media_subpixel = valori_subpixel.mean()
media_normal = valori_normal.mean()
media_halcon = valori_halcon.mean()
# X: numeri da 1 a N
x_sub = range(1, len(valori_subpixel) + 1)
x_norm = range(1, len(valori_normal) + 1)
x_halcon = range(1, len(valori_halcon) + 1)


# Estrai colonna 10 (RILEVAZIONE EFFETTUATA)
dist_mis_sub = df_subpixel.iloc[:, 10].abs()
dist_mis_norm = df_normal.iloc[:, 10].abs()
dist_mis_halcon = df_halcon.iloc[:,10].abs()
# Calcola le medie
media_dist_subpixel = dist_mis_sub.mean()
media_dist_normal = dist_mis_norm.mean()
media_dist_halcon = dist_mis_halcon.mean()
# X: numeri da 1 a N
x_dist_sub = range(1, len(dist_mis_sub) + 1)
x_dist_norm = range(1, len(dist_mis_norm) + 1)
x_dist_halcon = range(1, len(dist_mis_halcon) + 1)


# PRIMO GRAFICO - RIPETIBILITà SU CAMPINI A 0; Halcon vs. ChArUco (normale)
# # --- Configura figure e assi spezzati ---
fig, ax_top = plt.subplots(figsize=(20, 6))  # solo un asse
fig.subplots_adjust(hspace=0.1)

# --- Plotta i dati su entrambi gli assi ---
ax_top.plot(x_norm, valori_normal, marker='o', linestyle='-', color='#d39039', label='ChArUco', linewidth=1)
ax_top.plot(x_halcon, valori_halcon, marker='o', linestyle='-', color='#91d64d', label='Halcon', linewidth=1)

# ax_bottom.plot(x_norm, valori_normal, marker='o', linestyle='-', color='#d39039', linewidth=1)
# ax_bottom.plot(x_halcon, valori_halcon, marker='o', linestyle='-', color='#91d64d', linewidth=1)

# --- Linee medie (solo se stanno in uno dei due assi) ---
ax_top.axhline(y=media_normal, color='#d39039', linestyle='--', linewidth=1, label=f'ChArUco: {media_normal:.4f}')
ax_top.axhline(y=media_halcon, color='#91d64d', linestyle='--', linewidth=1, label=f'Halcon: {media_halcon:.4f}')

# --- Limiti degli assi ---
# ax_bottom.set_ylim(0.2, 0.25)
# ax_top.set_ylim(0.75, 0.8)

# --- Nascondi spines per creare effetto “spezzato” ---
# ax_top.spines['bottom'].set_visible(False)
# ax_bottom.spines['top'].set_visible(False)
# ax_top.tick_params(labeltop=False)
# ax_bottom.xaxis.tick_bottom()

# --- Aggiungi diagonali ---
# d = 0.001  # dimensione delle diagonali
# kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
# ax_top.plot((-d, +d), (-d, +d), **kwargs)
# ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

# kwargs.update(transform=ax_bottom.transAxes)
# ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
# ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# --- Etichette ---
ax_top.set_xlabel("Campioni")
ax_top.set_ylabel("Errore (mm)")
ax_top.set_title("Variazione dell'errore - Ripetibilità")
ax_top.legend()
ax_top.grid(True, linestyle=':', linewidth=0.5)
# ax_top.grid(True, linestyle=':', linewidth=0.5)

# SECONDO GRAFICO - RIPETIBILITà SU CAMPINI A 0; ChArUco (normale) vs ChArUco (subpixel refinement)
plt.figure(figsize=(20, 5))
plt.plot(x_sub, valori_subpixel, marker='o', linestyle='-', color='#3976d3', label='ChArUco SubPixel', linewidth=1)
plt.plot(x_norm, valori_normal, marker='o', linestyle='-', color='#d39039', label='ChArUco', linewidth=1)

# Linee della media
plt.axhline(y=media_subpixel, color='#3976d3', linestyle='--', linewidth=1, label=f'ChArUco SubPixel: {media_subpixel:.4f}')
plt.axhline(y=media_normal, color='#d39039', linestyle='--', linewidth=1, label=f'ChArUco: {media_normal:.4f}')

# Etichette e stile
plt.xlabel("Campioni")
plt.ylabel("Error (mm)")
plt.title("ChArUco vs ChArUco SubPixel")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend()

# Mostra il grafico
plt.tight_layout()
plt.show()


# TERZO GRAFICO - RIPETIBILITà SU CAMPINI A 0; VALORE MISURATO Halcon vs. ChArUco (normale)
plt.figure(figsize=(20, 5))
plt.plot(x_dist_norm,  dist_mis_norm, marker='o', linestyle='-', color='#d39039', label='ChArUco', linewidth=1)
plt.plot(x_dist_halcon, dist_mis_halcon, marker='o', linestyle='-', color='#91d64d', label='Halcon', linewidth=1)

# Linee della media
plt.axhline(y=media_dist_normal, color='#d39039', linestyle='--', linewidth=1, label=f'ChArUco: {media_dist_normal:.4f}')
plt.axhline(y=media_dist_halcon, color='#91d64d', linestyle='--', linewidth=1, label=f'Halcon: {media_dist_halcon:.4f}')
plt.axhline(y=130.8625232, color='red', linestyle='--', linewidth=1, label=f'Real Value')


# Etichette e stile
plt.xlabel("Campioni")
plt.ylabel("Stima (mm)")
plt.title("Variazione della Misura - Ripetibilità")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend()

# Mostra il grafico
plt.tight_layout()
plt.show()


#--- GRAFICO 3 DEVIAZIONE STANDARD E SUA VARIAIZONE CHARUCO VS HALCON
# Funzione per calcolare std per ciascun file
def calcola_std_per_file(file_list):
    std_list = []
    for f in file_list:
        df = pd.read_csv(f)
        col = df.iloc[:, 11].abs()
        std = np.std(col)
        std_list.append(std)
    return np.array(std_list)

# Percorsi ai file
file_charuco = glob("../output/set_*_charuco.csv")
file_charuco_sub = glob("../output/set_*_charuco_sub.csv")
file_halcon = glob("../output/set_*_halcon.csv")

# Calcola std per ogni file
std_charuco_vals = calcola_std_per_file(file_charuco)
std_charuco_sub_vals = calcola_std_per_file(file_charuco_sub)
std_halcon_vals = calcola_std_per_file(file_halcon)

# Calcola la media e l'intervallo [min, max] per yerr
mean_charuco = np.mean(std_charuco_vals)
mean_charuco_sub = np.mean(std_charuco_sub_vals)
mean_halcon = np.mean(std_halcon_vals)

err_charuco = [mean_charuco - np.min(std_charuco_vals), np.max(std_charuco_vals) - mean_charuco]
err_charuco_sub = [mean_charuco_sub - np.min(std_charuco_sub_vals), np.max(std_charuco_sub_vals) - mean_charuco_sub]
err_halcon = [mean_halcon - np.min(std_halcon_vals), np.max(std_halcon_vals) - mean_halcon]


# Prepara i dati per il grafico
metodi = ['ChArUco', 'ChArUco Sub', 'Halcon']
dev_std = [mean_charuco, mean_charuco_sub, mean_halcon]
yerr = np.array([err_charuco, err_charuco_sub, err_halcon]).T  # transpose per matplotlib

# Costruisci le etichette della legenda con i valori incorporati
labels = [
    f"ChArUco: {mean_charuco:.3f}",
    f"ChArUco Sub: {mean_charuco_sub:.3f}",
    f"Halcon: {mean_halcon:.3f}"
]

error_kw = dict(
    lw=1,      # spessore linea error bar
    capsize=5, # lunghezza alette
    ecolor='red'
)

# Crea la figura
plt.figure(figsize=(10, 6))

# Disegna le barre con yerr
bars = plt.bar(
    metodi,
    dev_std,
    yerr=yerr,
    color=['#d39039', '#3976d3', '#91d64d'],
    error_kw=error_kw
)

# Aggiungi legenda usando le barre come handles
plt.legend(bars, labels, title="Metodo (Dev. Std.)")
plt.ylabel("Variazione (mm)")
plt.title("Stima della stabilità per metodo")
plt.grid(True, axis='y', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.show()