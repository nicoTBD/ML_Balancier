import pandas as pd
import matplotlib.pyplot as plt
import os

def normalize_data(data):
    # Normalisation des données entre -1 et 1
    min_val = data.min()
    max_val = data.max()
    normalized_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    return normalized_data

def label_data(acc_data, confidence_interval_low, confidence_interval_high):
    # Labelisation des données en fonction des bornes de confiance
    labels = []
    step = 45
    half_step = step/2
    for i in range(0, len(acc_data), step):
        window_data = acc_data[i:i+step]
        majority_count = sum((window_data > confidence_interval_high) | (window_data < confidence_interval_low))
        label = 1 if majority_count >= half_step else 0
        labels.extend([label] * step)
    return labels

def process_axis(data, axis_name, confidence_interval_low, confidence_interval_high, output_folder, ax):
    # Normaliser les données de l'accéléromètre sur l'axe spécifié
    acc_data = normalize_data(data[f'Acc{axis_name} [mg]'])

    # Labeliser les données
    labels = label_data(acc_data, confidence_interval_low, confidence_interval_high)

    # Ajouter les étiquettes au DataFrame
    data[f'Label{axis_name}'] = labels

    # Stocker les données labelisées dans des listes séparées
    labeled_1_data = data[data[f'Label{axis_name}'] == 1]
    labeled_0_data = data[data[f'Label{axis_name}'] == 0]

    # Enregistrer les données labelisées dans des fichiers CSV
    labeled_1_data.to_csv(os.path.join(output_folder, f'labelised{axis_name}1.csv'), index=False)
    labeled_0_data.to_csv(os.path.join(output_folder, f'labelised{axis_name}0.csv'), index=False)

    # Afficher les données labelisées sur le graphique
    ax.scatter(data['T [ms]'], acc_data, c=data[f'Label{axis_name}'], cmap='bwr', marker='o', label=f'Acc{axis_name}')
    ax.set_xlabel('Temps (T) en ms')
    ax.set_ylabel(f'Accélération axe {axis_name} normalisée en mg')
    ax.set_title(f'Graphique de l\'accéléromètre axe {axis_name} avec labelisation')
    ax.grid(True)

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    data = pd.read_csv('../Projet_Balancier/input_dataset_concatenate/donnees_concatenes.csv')

    # Créer un dossier pour stocker les fichiers de sortie
    output_folder = 'outputLabelised'
    os.makedirs(output_folder, exist_ok=True)

    # Définir les bornes de confiance pour chaque axe
    confidence_intervals = {
        'X': (0.18, 0.26),
        'Y': (0.28, 0.38),
        'Z': (0.06, 0.14)
    }

    # Créer une seule fenêtre avec trois sous-graphiques
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Traiter chaque axe et afficher les données labelisées
    for i, (axis_name, confidence_interval) in enumerate(confidence_intervals.items()):
        process_axis(data, axis_name, *confidence_interval, output_folder, axs[i])

    # Ajouter une légende commune pour les trois graphiques
    axs[0].legend(loc='upper right')

    # Afficher le graphique
    plt.tight_layout()
    plt.show()
