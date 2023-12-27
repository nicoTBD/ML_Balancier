import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import mplcursors
from ipywidgets import interact

def normalize_data(data):
    # Normalisation des données entre -1 et 1
    min_val = data.min()
    max_val = data.max()
    normalized_data = -1 + 2 * (data - min_val) / (max_val - min_val)
    return normalized_data

def label_data(acc_data, confidence_interval_low, confidence_interval_high):
    # Labelisation des données en fonction des bornes de confiance
    labels = []
    step = 72
    half_step = 0.4 * step
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

    # Supprimer les données avant T=10000 ms et au-delà de 221000 ms
    data = data[(data['T [ms]'] >= 10000) & (data['T [ms]'] <= 221000)].copy()

    # Décaler les données au-delà de 59274 vers le nouvel offset de 32594
    data.loc[data['T [ms]'] > 59274, 'T [ms]'] += 32594 - 59274

    # Décaler les données au-delà de 179917 vers le nouvel offset de 148620
    data.loc[data['T [ms]'] > 179917, 'T [ms]'] += 148620 - 179917

    # Stocker les données labelisées dans des listes séparées
    labeled_1_data = data[data[f'Label{axis_name}'] == 1]
    labeled_0_data = data[data[f'Label{axis_name}'] == 0]

    # Enregistrer les données labelisées dans des fichiers CSV
    labeled_1_data.to_csv(os.path.join(output_folder, f'labelised{axis_name}1.csv'), index=False)
    labeled_0_data.to_csv(os.path.join(output_folder, f'labelised{axis_name}0.csv'), index=False)

    # Afficher les données labelisées sur le graphique
    ax.scatter(data['T [ms]'], acc_data, c=data[f'Label{axis_name}'], cmap='bwr', marker='o')
    ax.set_xlabel('Temps (T) en ms')
    ax.set_ylabel(f'Accélération axe {axis_name} en mg normalisée')
    ax.set_title(f'Graphique de l\'accéléromètre axe {axis_name} avec labelisation')
    ax.grid(True)

    # Ajouter une légende commune pour les trois graphiques
    legend_labels = {0: 'statique', 1: 'en mouvement'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for
               color, label in zip(['blue', 'red'], legend_labels.values())]
    ax.legend(handles=handles, loc='upper right', title='Légende')

def plot_acceleration_3d(data, ax):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['T [ms]'], data['AccX [mg]'], data['AccY [mg]'], c=data['LabelX'], cmap='bwr', marker='o')
    ax.set_xlabel('Temps (T) en ms')
    ax.set_ylabel('Accélération axe X en mg normalisée')
    ax.set_zlabel('Accélération axe Y en mg normalisée')
    ax.set_title('Graphique de l\'accéléromètre en 3D avec labelisation')
    ax.grid(True)

def plot_data_3d_with_slider(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normaliser les données de l'accéléromètre
    normalized_data = normalize_data(data[['AccX [mg]', 'AccY [mg]', 'AccZ [mg]']])

    # Afficher le premier point sur les 3 axes
    scatter = ax.scatter(
        normalized_data['AccX [mg]'].iloc[0],
        normalized_data['AccY [mg]'].iloc[0],
        normalized_data['AccZ [mg]'].iloc[0],
        c='red',
        marker='o',
        label='Point Actuel'
    )

    # Configurer les axes pour qu'ils aillent de -1 à 1
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('AccX [mg]')
    ax.set_ylabel('AccY [mg]')
    ax.set_zlabel('AccZ [mg]')
    ax.set_title('Représentation 3D des données avec curseur temporel')
    ax.legend()

    global slider  # Définir le curseur en tant que variable globale
    ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Temps', 0, len(data) - 1, valinit=0, valstep=1)

    def update_plot(val):
        time_index = int(slider.val)
        x = normalized_data['AccX [mg]'].iloc[time_index]
        y = normalized_data['AccY [mg]'].iloc[time_index]
        z = normalized_data['AccZ [mg]'].iloc[time_index]

        # Utiliser set_offsets pour mettre à jour les données du nuage de points
        scatter.set_offsets([[x, y, z]])

        # Mettre à jour l'étiquette du point actuel
        scatter._offsets3d = ([x], [y], [z])

        # Forcer le réaffichage de la figure
        plt.draw()

    slider.on_changed(update_plot)

    plt.show()


if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    data = pd.read_csv('input_dataset_concatenate/donnees_concatenes.csv')

    # Prétraitement des données
    data = data[(data['T [ms]'] >= 10000) & (data['T [ms]'] <= 221000)]
    data.loc[(data['T [ms]'] >= 32593) & (data['T [ms]'] <= 59274), 'T [ms]'] += 1

    # Créer un dossier pour stocker les fichiers de sortie
    output_folder = 'outputLabelisedXYZ'
    os.makedirs(output_folder, exist_ok=True)

    # Définir les bornes de confiance pour chaque axe
    confidence_intervals = {
        'X': (0.18, 0.25),
        'Y': (0.28, 0.36),
        'Z': (0.092, 0.126)
    }

    # Créer une seule fenêtre avec trois sous-graphiques
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Traiter chaque axe et afficher les données labelisées
    for i, (axis_name, confidence_interval) in enumerate(confidence_intervals.items()):
        process_axis(data, axis_name, *confidence_interval, output_folder, axs[i])

    # Sauvegarder toutes les données labellisées dans un seul fichier CSV
    labeled_data_all_axes = pd.concat([data[data['LabelX'] == 0], data[data['LabelX'] == 1],
                                      data[data['LabelY'] == 0], data[data['LabelY'] == 1],
                                      data[data['LabelZ'] == 0], data[data['LabelZ'] == 1]])

    # Afficher les données en 3D avec un curseur temporel
    plot_data_3d_with_slider(data)

    # Afficher le graphique en 3D
    plot_acceleration_3d(data, ax=None)

    # Afficher le graphique
    plt.tight_layout()
    plt.show()
