import pandas as pd
import os

# Dossier contenant les fichiers CSV
dossier_input = 'input_dataset_13_12/'

# Liste des noms de fichiers CSV
noms_fichiers = ['SensorTile_Log_N000.csv', 'SensorTile_Log_N001.csv', 'SensorTile_Log_N002.csv', 'SensorTile_Log_N003.csv']

# Initialiser une liste pour stocker les DataFrames de chaque fichier
list_df = []

# Lire chaque fichier CSV et ajouter son DataFrame à la liste
for nom_fichier in noms_fichiers:
    chemin_fichier = os.path.join(dossier_input, nom_fichier)
    df = pd.read_csv(chemin_fichier)
    list_df.append(df)

# Concaténer les DataFrames dans un seul DataFrame
donnees_concat = pd.concat(list_df, ignore_index=True)

# Sauvegarder le DataFrame concaténé dans un nouveau fichier CSV
donnees_concat.to_csv('donnees_concatenes.csv', index=False)
