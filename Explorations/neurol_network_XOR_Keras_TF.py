# Inpired from correction.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
import timeit

# Charger les données à partir du fichier CSV
donnees = pd.read_csv('venv/input/SensorTile_Log_N005.csv')

# Prétraitement des données
# Assurez-vous que vos colonnes d'entrée sont adaptées au modèle
E = donnees[['feature1', 'feature2']].values
# Assurez-vous que votre colonne de sortie est adaptée au modèle
Y = donnees['label'].values

# create model
model = Sequential()
model.add(Dense(100, input_dim=2))
model.add(Dense(200, 'sigmoid'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
N_iteration = 1000

start_time = timeit.default_timer()
history = model.fit(E.transpose(), Y, epochs=N_iteration, verbose=0, batch_size=100)
print("Temps d'entrainement: ", timeit.default_timer() - start_time) # le résultat est en µs

# plot metrics
# Créer une seule figure avec quatre sous-graphiques
plt.figure(figsize=(12, 10))

# Premier sous-graphique : Evolution de l'erreur quadratique moyenne
plt.subplot(2, 2, 1)
plt.title('Evolution de l''erreur quadratique moyenne')
plt.plot(history.history['mean_squared_error'])


# Deuxième sous-graphique : Evolution de l'erreur absolue moyenne
plt.subplot(2, 2, 2)
plt.title('Evolution de l''erreur absolue moyenne')
plt.plot(history.history['mean_absolute_error'])

# Troisième sous-graphique : Erreur moyenne absolue en pourcentage
plt.subplot(2, 2, 3)
plt.title('Erreur moyenne absolue en pourcentage')
plt.plot(history.history['mean_absolute_percentage_error'])

# Quatrième sous-graphique : Proximité cosinus
plt.subplot(2, 2, 4)
plt.title('Proximité cosinus')
plt.plot(history.history['cosine_proximity'])

# Ajustement automatique de la disposition des sous-graphiques
plt.tight_layout()

# Afficher la figure
plt.show()