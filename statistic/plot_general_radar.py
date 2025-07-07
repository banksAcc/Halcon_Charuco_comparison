import numpy as np
import matplotlib.pyplot as plt

# 1) Dati di esempio (raw, senza normalizzazione)
data ={'ChArUco': {'dev std acc': 0.126,
             'dev std rip': 0.35,
             'err acc': 0.12975000000000003,
             'err rip': 0.22760000000000002,
             'speed': 0.6950000000000001},
 'Halcon': {'dev std acc': 0.846,
            'dev std rip': 0.75,
            'err acc': 0.86565,
            'err rip': 0.772,
            'speed': 0.26999999999999996}
}

params = [
    'speed',
    'err rip',
    'dev std rip',
    'err acc',
    'dev std acc'
]

systems = list(data.keys())

# Costruisci un array n_sistemi x n_parametri
values = np.array([[data[s][p] for p in params] for s in systems])

# Calcola il valore massimo per impostare i limiti del radar
max_val = values.max()

# Angoli del radar
N = len(params)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # chiudi il cerchio

# Colori e stili
colors = ['#d39039', '#3976d3']
line_styles = ['-', '--']

# Crea il plot
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

ax.set_xticklabels(
    params,
    fontsize=9,
    fontweight='bold'   # <–– qui
)

labels = ax.get_xticklabels()
for lbl in labels:
    # portala in primo piano
    lbl.set_zorder(100)
    # box colorato dietro il testo
    lbl.set_bbox({
        'facecolor' : "#ace3f9",   # bianco (o qualsiasi colore tu voglia)
        'edgecolor' : "#ace3f9",   # bordo della box
        'boxstyle'  : 'round,pad=0.3',
        'alpha'     : 0.8
    })

for i, sys in enumerate(systems):
    vals = values[i].tolist()
    vals += vals[:1]
    # fill con alpha per distinguere
    ax.fill(angles, vals,
            facecolor=colors[i], alpha=0.25, edgecolor=colors[i], linewidth=2)
    ax.plot(angles, vals,
            color=colors[i], linestyle=line_styles[i], linewidth=2,
            label=f"{sys}")

# Etichette
ax.set_xticks(angles[:-1])
ax.set_xticklabels(params, fontsize=10)

ax.set_yticklabels([])

# Limiti radiali
ax.set_ylim(0, max_val * 1.1)

# Griglia radiale
ax.yaxis.set_tick_params(labelsize=8)
ax.grid(True, linestyle=':', linewidth=0.5)

# Legenda
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

ax.set_title("Halcon vs ChArUco General View", va='bottom')

plt.tight_layout()
plt.show()
