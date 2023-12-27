import pandas as pd
import matplotlib.pyplot as plt


def labeliser_mouvement(acceleration_data, confidence_interval=(0.14, 0.31), majority_threshold=0.5):
    """
    Labélisation des données d'accéléromètre pour déterminer le mouvement du pendule.

    Parameters:
        - acceleration_data: Données d'accéléromètre sur l'axe X
        - confidence_interval: Bornes de confiance pour déterminer le mouvement statique
        - majority_threshold: Seuil de majorité pour attribuer l'étiquette 1 ou 0

    Returns:
        - Liste d'étiquettes (0 ou 1) pour chaque groupe de 10 données
    """
    labels = []
    step = 10

    for i in range(0, len(acceleration_data), step):
        subset = acceleration_data[i:i + step]

        # Normalisation des données entre -1 et 1
        normalized_data = 2 * (subset - subset.min()) / (subset.max() - subset.min()) - 1

        # Vérification de la majorité dépassant les bornes de confiance
        majority_condition = ((normalized_data >= confidence_interval[0]) & (
                    normalized_data <= confidence_interval[1])).sum() >= majority_threshold * step

        # Attribution de l'étiquette en fonction de la condition de majorité
        label = 1 if majority_condition else 0
        labels.extend([label] * len(subset))  # Ajustement de la longueur

    return labels


if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pd.read_csv('venv/input/SensorTile_Log_N005.csv')

    # Appliquer la labélisation des données d'accéléromètre
    donnees['AccX [mg]'] = 2 * (donnees['AccX [mg]'] - donnees['AccX [mg]'].min()) / (
                donnees['AccX [mg]'].max() - donnees['AccX [mg]'].min()) - 1
    labels = labeliser_mouvement(donnees['AccX [mg]'])

    # Créer un graphique pour l'accéléromètre avec les étiquettes
    plt.figure(figsize=(12, 6))
    plt.plot(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX', color='blue')
    plt.scatter(donnees['T [ms]'], donnees['AccX [mg]'], c=labels, cmap='coolwarm', marker='o', label='Mouvement détecté')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération normalisée')
    plt.title('Graphique de l\'accéléromètre avec détection de mouvement')
    plt.legend()

    plt.grid(True)
    plt.show()
