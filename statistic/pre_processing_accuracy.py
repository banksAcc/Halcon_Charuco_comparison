import pandas as pd
import numpy as np
from glob import glob

# Configurazione sistemi e path
systems = {
    'charuco':    "../output/set_{shift}_charuco.csv",
    'charuco_sub':"../output/set_{shift}_charuco_sub.csv",
    'halcon':     "../output/set_{shift}_halcon.csv"
}
shifts = ['0', '+10', '-10']

# Definizione delle coppie e delle distanze reali
pairs = [
    ('0',  '+10', 100),
    ('0',  '-10', 100),
    ('-10','+10', 200)
]

# Funzione per calcolare la distanza Euclidea di un marker
def compute_distances(df):
    # Marker 1 colonne
    df['dist_marker_1'] = np.sqrt(
        df['M1_tx_mm']**2 + df['M1_ty_mm']**2 + df['M1_tz_mm']**2
    )
    # Marker 2 colonne
    df['dist_marker_2'] = np.sqrt(
        df['M2_tx_mm']**2 + df['M2_ty_mm']**2 + df['M2_tz_mm']**2
    )
    return df

# Loop su ciascun sistema
for system_name, template in systems.items():
    # Carica e processa i tre DataFrame
    dfs = {}
    for shift in shifts:
        path = template.format(shift=shift)
        df = pd.read_csv(path)
        df = compute_distances(df)
        dfs[shift] = df

    # Costruisci lista di righe per il DataFrame di output
    rows = []
    for a, b, true_dist in pairs:
        df_a = dfs[a]; df_b = dfs[b]
        # Assumiamo stessa lunghezza e allineamento dei campioni
        for marker in [1, 2]:
            col = f'dist_marker_{marker}'
            dist_a = df_a[col].values
            dist_b = df_b[col].values
            # Differenza per campione
            delta = np.abs(dist_b - dist_a)
            # Errore assoluto rispetto a true_dist per campione
            error = np.abs(delta - true_dist)
            # Aggiungi ogni campione come riga
            for i, (d1, d2, d, e) in enumerate(zip(dist_a, dist_b, delta, error), start=1):
                rows.append({
                    'pair':         f'{a}_{b}',
                    'marker':       marker,
                    'sample_idx':   i,
                    'dist1_mm':     d1,
                    'dist2_mm':     d2,
                    'delta_mm':     d,
                    'true_mm':      true_dist,
                    'error_mm':     e
                })

    # Crea DataFrame finale e salva CSV
    out_df = pd.DataFrame(rows)
    out_filename = f'accuracy_{system_name}.csv'
    out_df.to_csv(out_filename, index=False)
    print(f"Saved accuracy results for {system_name} to {out_filename}")
