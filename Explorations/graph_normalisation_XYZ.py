import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pd.read_csv('../Projet_Balancier/input_dataset_concatenate/donnees_concatenes.csv')

    # Normaliser les données entre -1 et 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    donnees[['AccX [mg]', 'AccY [mg]', 'AccZ [mg]', 'GyroX [mdps]', 'GyroY [mdps]', 'GyroZ [mdps]']] = scaler.fit_transform(
        donnees[['AccX [mg]', 'AccY [mg]', 'AccZ [mg]', 'GyroX [mdps]', 'GyroY [mdps]', 'GyroZ [mdps]']])

    # Créer un graphique pour l'accéléromètre et le gyroscope
    plt.figure(figsize=(12, 12))

    # Sous-graphique 1 : Accélération sur l'axe X, Y, Z
    plt.subplot(3, 1, 1)
    plt.scatter(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX', marker='.')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération normalisée en X')
    plt.title('Accélération sur l\'axe X en fonction de T')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.scatter(donnees['T [ms]'], donnees['AccY [mg]'], label='AccY', marker='.')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération normalisée en Y')
    plt.title('Accélération sur l\'axe Y en fonction de T')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(donnees['T [ms]'], donnees['AccZ [mg]'], label='AccZ', marker='.')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération normalisée en Z')
    plt.title('Accélération sur l\'axe Z en fonction de T')
    plt.legend()
    plt.grid(True)

    # Ajustement automatique de la disposition des sous-graphiques
    plt.tight_layout()

    # Afficher la figure
    plt.show()
