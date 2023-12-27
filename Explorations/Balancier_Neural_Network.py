import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import timeit
import matplotlib.pyplot as plt

# Charger les données du capteur du pendule
data = pd.read_csv('venv/outputLabelisedXYZ/labelised_all_axes.csv')

# Convertir les labels en une seule colonne pour chaque axe (X, Y, Z)
label_encoder = LabelEncoder()
data['LabelX'] = label_encoder.fit_transform(data['LabelX'])
data['LabelY'] = label_encoder.fit_transform(data['LabelY'])
data['LabelZ'] = label_encoder.fit_transform(data['LabelZ'])

# Diviser les données en entrées (X) et sorties (Y)
X = data[['AccX [mg]', 'AccY [mg]', 'AccZ [mg]']].values
Y = data[['LabelX', 'LabelY', 'LabelZ']].values

# Data cleaning, need to transpose X
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Binariser les labels à l'aide de MultiLabelBinarizer
mlb = MultiLabelBinarizer()
Y_train_binary = mlb.fit_transform(map(tuple, Y_train))
Y_test_binary = mlb.transform(map(tuple, Y_test))

# Créer le modèle
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(3, activation='sigmoid'))
opt = optimizers.Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Entraînement du modèle
N_iteration = 150
start_time = timeit.default_timer()
history = model.fit(Y_train_binary, Y_train_binary, validation_split=0.33, epochs=N_iteration, batch_size=10, verbose=1)
print("Temps passé : %.2fs" % (timeit.default_timer() - start_time))

# Évaluation du modèle
scores = model.evaluate(X_test, Y_test_binary)
print("\nEvaluation sur le test data - Loss: %.4f - Accuracy: %.2f%% " % (scores[0], scores[1] * 100))

# Visualisation des résultats
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
