import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pd.read_csv('venv/input/SensorTile_Log_N005.csv')

    # Créer une figure avec trois sous-graphiques
    plt.figure(figsize=(12, 12))

    # Sous-graphique 1 : Données d'accélération sur l'axe X
    plt.subplot(3, 1, 1)
    plt.plot(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération en mg sur l\'axe X')
    plt.title('Données d\'accélération sur l\'axe X en fonction du temps')
    plt.legend()
    plt.grid(True)

    # Sous-graphique 2 : FFT de l'accélération sur l'axe X
    plt.subplot(3, 1, 2)
    signal = donnees['AccX [mg]']
    N = len(signal)
    T = donnees['T [ms]'].iloc[1] - donnees['T [ms]'].iloc[0]  # Période d'échantillonnage
    freq = fftfreq(N, d=T)[:N//2]
    fft_vals = fft(signal)
    fft_vals = 2.0/N * np.abs(fft_vals[0:N//2])
    plt.plot(freq, fft_vals, label='FFT AccX')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT de l\'accélération sur l\'axe X')
    plt.legend()
    plt.grid(True)

    # Sous-graphique 3 : DFT de l'accélération sur l'axe X
    plt.subplot(3, 1, 3)
    dft_vals = np.fft.fft(signal)
    plt.plot(np.abs(dft_vals.imag), label='DFT AccX')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.title('DFT de l\'accélération sur l\'axe X')
    plt.legend()
    plt.grid(True)

    # Ajustement automatique de la disposition des sous-graphiques
    plt.tight_layout()

    # Afficher la figure
    plt.show()
