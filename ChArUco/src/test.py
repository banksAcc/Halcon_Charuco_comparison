import numpy as np

# Carica il file .npz
data = np.load('file.npz')

# Itera su tutte le chiavi e stampa nome + contenuto
for key in data.files:
    print(f"🗂 Array: {key}")
    print(data[key])
    print("-" * 40)

# (opzionale) chiudi il file
data.close()
