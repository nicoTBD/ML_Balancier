import pandas as pd
import matplotlib.pyplot as plt
import os

def normalize_data(data):
    # Normalisation des données entre -1 et 1
    min_val = data.min()
    max_val = data.max()
    normalized_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    return normalized_data

def label_data(accX_data, confidence_interval_low, confidence_interval_high):
    # Labelisation des données en fonction des bornes de confiance
    labels = []
    step = 45
    half_step = step/2
    for i in range(0, len(accX_data), step):
        window_data = accX_data[i:i+step]
        majority_count = sum((window_data > confidence_interval_high) | (window_data < confidence_interval_low))
        label = 1 if majority_count >= half_step else 0
        labels.extend([label] * step)
    return labels

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    data = pd.read_csv('../Projet_Balancier/input_dataset_concatenate/donnees_concatenes.csv')

    # Normaliser les données de l'accéléromètre sur l'axe X
    accX_data = normalize_data(data['AccX [mg]'])

    # Définir les bornes de confiance
    confidence_interval_low = 0.18
    confidence_interval_high = 0.26

    # Labeliser les données
    labels = label_data(accX_data, confidence_interval_low, confidence_interval_high)

    # Ajouter les étiquettes au DataFrame
    data['Label'] = labels

    # Stocker les données labelisées dans des listes séparées
    labeled_1_data = data[data['Label'] == 1]
    labeled_0_data = data[data['Label'] == 0]

    # Créer un dossier pour stocker les fichiers de sortie
    output_folder = 'outputLabelised'
    os.makedirs(output_folder, exist_ok=True)

    # Enregistrer les données labelisées dans des fichiers CSV
    labeled_1_data.to_csv(os.path.join(output_folder, 'labelisedX1.csv'), index=False)
    labeled_0_data.to_csv(os.path.join(output_folder, 'labelisedX0.csv'), index=False)

    # Créer un graphique pour l'accéléromètre avec les points labelisés
    plt.scatter(data['T [ms]'], accX_data, c=data['Label'], cmap='bwr', marker='o')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération axe X normalisée en mg')
    plt.title('Graphique de l\'accéléromètre avec labelisation')
    plt.colorbar(ticks=[0, 1], label='Label')
    plt.grid(True)

    # Afficher le graphique
    plt.tight_layout()  # Ajustement automatique de la disposition des sous-graphiques
    plt.show()
