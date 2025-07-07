# 1) I tuoi dati raw
data = {
    'Halcon': {
        'speed':            0.784,
        'err rip':          0.2280,
        'dev std rip':      0.5,
        'err acc':          0.2687,
        'dev std acc':      0.154
    },
    'ChArUco': {
        'speed':            0.444,
        'err rip':          0.7724,
        'dev std rip':      1.3,
        'err acc':          1.7405,
        'dev std acc':      0.874
    }
}

# 2) Definisci hard-coded i range (minimo, massimo) su cui normalizzare
#    (Ti basta mettere i valori estremi che ti interessano,
#     non per forza quelli dei tuoi soli due sistemi.)
ranges = {
    'speed':       (0.2, 1),   # es. tempi da 0.5s (ottimo) a 1.5s (peggiore)
    'err rip':     (0.0, 1.0),   # errori di ripetibilità da 0 a 1 mm
    'dev std rip': (0.0, 2.0),   # dev std ripetibilità da 0 a 2 mm
    'err acc':     (0.0, 2.0),   # errori di accuratezza da 0 a 2 mm
    'dev std acc': (0.0, 1.0)    # dev std accuratezza da 0 a 1 mm
}

# 3) Funzione di normalizzazione “più piccolo è meglio”
def normalize_min_better(value, vmin, vmax):
    """Mappa [vmin..vmax] → [1..0], clip."""
    # clip per sicurezza
    v = max(min(value, vmax), vmin)
    return (vmax - v) / (vmax - vmin)

# 4) Applichiamo la normalizzazione
normalized = {}
for system, metrics in data.items():
    normalized[system] = {}
    for name, raw in metrics.items():
        vmin, vmax = ranges[name]
        norm = normalize_min_better(raw, vmin, vmax)
        normalized[system][name] = norm

# 5) Guarda il risultato
from pprint import pprint
pprint(normalized)
