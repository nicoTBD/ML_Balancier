import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pd.read_csv('input_dataset_concatenate/donnees_concatenes.csv')

    # Normaliser les données entre -1 et 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    donnees[['AccX [mg]', 'AccY [mg]', 'AccZ [mg]']] = scaler.fit_transform(
        donnees[['AccX [mg]', 'AccY [mg]', 'AccZ [mg]']])

    # Créer un graphique pour les histogrammes de l'accélération
    plt.figure(figsize=(12, 4))

    # Histogramme de l'accélération sur l'axe X
    plt.subplot(1, 3, 1)
    plt.hist(donnees['AccX [mg]'], bins=100, color='blue', edgecolor='black')
    plt.axhline(y=400, color='r', linestyle='--')  # Seuil horizontal à 1000 Hz
    plt.xlabel('Accélération normalisée en X')
    plt.ylabel('Fréquence')
    plt.title('Répartition de l\'accélération sur l\'axe X')

    # Histogramme de l'accélération sur l'axe Y
    plt.subplot(1, 3, 2)
    plt.hist(donnees['AccY [mg]'], bins=100, color='green', edgecolor='black')
    plt.axhline(y=400, color='r', linestyle='--')  # Seuil horizontal à 1000 Hz
    plt.xlabel('Accélération normalisée en Y')
    plt.ylabel('Fréquence')
    plt.title('Répartition de l\'accélération sur l\'axe Y')

    # Histogramme de l'accélération sur l'axe Z
    plt.subplot(1, 3, 3)
    plt.hist(donnees['AccZ [mg]'], bins=100, color='red', edgecolor='black')
    plt.axhline(y=400, color='r', linestyle='--')  # Seuil horizontal à 1000 Hz
    plt.xlabel('Accélération normalisée en Z')
    plt.ylabel('Fréquence')
    plt.title('Répartition de l\'accélération sur l\'axe Z')

    # Ajustement automatique de la disposition des sous-graphiques
    plt.tight_layout()

    # Afficher la figure
    plt.show()
