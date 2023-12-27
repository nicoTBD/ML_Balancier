import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pd.read_csv('input_dataset_concatenate/donnees_concatenes.csv')

    # Créer un graphique pour l'accéléromètre
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)  # Deux lignes, une colonne, premier graphique
    plt.plot(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX')
    plt.plot(donnees['T [ms]'], donnees['AccY [mg]'], label='AccY')
    plt.plot(donnees['T [ms]'], donnees['AccZ [mg]'], label='AccZ')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération en mg')
    plt.title('Graphique de l\'accéléromètre en fonction de T')
    plt.legend()
    plt.grid(True)

    # Créer un graphique pour le gyroscope
    plt.subplot(2, 1, 2)  # Deux lignes, une colonne, deuxième graphique
    plt.plot(donnees['T [ms]'], donnees['GyroX [mdps]'], label='GyroX')
    plt.plot(donnees['T [ms]'], donnees['GyroY [mdps]'], label='GyroY')
    plt.plot(donnees['T [ms]'], donnees['GyroZ [mdps]'], label='GyroZ')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Vitesse angulaire en mdps')
    plt.title('Graphique du gyroscope en fonction de T')
    plt.legend()
    plt.grid(True)

    # Afficher les graphiques
    plt.tight_layout()  # Ajustement automatique de la disposition des sous-graphiques
    plt.show()