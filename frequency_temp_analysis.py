import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données avec Pandas
donnees = pd.read_csv('input_dataset_concatenate/donnees_concatenes.csv')

# Créer un graphique pour l'analyse temporelle
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)  # Deux lignes, une colonne, premier graphique
plt.plot(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX')
plt.xlabel('Temps (T) en ms')
plt.ylabel('Accélération en X (AccX) en mg')
plt.title('Analyse Temporelle de AccX')
plt.legend()
plt.grid(True)


# Analyse Fréquentielle
# Utilisation de la transformée de Fourier avec NumPy
signal = donnees['AccX [mg]'].values
freqs = np.fft.fftfreq(len(signal), d=(donnees['T [ms]'].iloc[1] - donnees['T [ms]'].iloc[0]) / 1000)
fft_vals = np.fft.fft(signal)

# Tracer le spectre de fréquence avec Matplotlib
plt.subplot(2, 1, 2)  # Deux lignes, une colonne, deuxième graphique
plt.plot(freqs, np.abs(fft_vals), label='Spectre de fréquence')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')
plt.title('Analyse Fréquentielle de AccX')
plt.legend()
plt.grid(True)

# Afficher les graphiques
plt.tight_layout()  # Ajustement automatique de la disposition des sous-graphiques
plt.show()
