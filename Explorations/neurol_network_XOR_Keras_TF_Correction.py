import timeit
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.python.framework.random_seed import set_random_seed

seed = 0
np.random.seed(seed)
set_random_seed(seed)

# Définition des exemples entrées / sorties
# Couples de valeurs en entrée :
# [0,0], [0.1,0], [0.2,0], ... [0.9,0], [1,0], [0,0.1], [0.1,0.1], ...
E1 = np.repeat(np.linspace(0, 1, 10).reshape([1, 10]), 10, axis=0)
E1 = E1.reshape(100)
E2 = np.repeat(np.linspace(0, 1, 10), 10)
E = np.array([E1, E2])

# Sorties correspondantes
# 0, 0, 0, ... 1, 1, 0, 0, ...
Y = np.zeros(E.shape[1])
for n_exe in range(0, E.shape[1]):
    if (E[0, n_exe] >= 0.5 > E[1, n_exe]) or (E[0, n_exe] < 0.5 <= E[1, n_exe]):
        Y[n_exe] = 1

# plt.figure()
# plt.plot(E[0,:])
# plt.plot(E[1,:])
# plt.plot(Y)
# plt.legend(('E1','E2','Y'))

# Data cleaning, need to transpose E
Et = E.transpose()
E_train, E_test, Y_train, Y_test = train_test_split(Et, Y, test_size=0.33, random_state=seed)

# create model
model = Sequential()
model.add(Dense(3, input_dim=2, activation='relu')) # 50, 2, relu
model.add(Dense(1, activation='sigmoid'))
opt = optimizers.SGD(lr=0.95, decay=0, momentum=0.2) # 0.9, 0, 0.2
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc']) # MSE, opt, acc
# print model in .png file
# plot_model(model)

# Entraînement du modèle
N_iteration = 400

# train
start_time = timeit.default_timer()
history = model.fit(E_train, Y_train, validation_split=0.15, shuffle=False, epochs=N_iteration, verbose=0, batch_size=10)
print("Temps passé : %.2fs" % (timeit.default_timer() - start_time))

# plot figures
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot accuracy
axs[0, 0].plot(history.history['acc'])
axs[0, 0].plot(history.history['val_acc'])
axs[0, 0].set_title('Model accuracy')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].legend(['Train', 'Val'], loc='upper left')

# Plot loss
axs[0, 1].plot(history.history['loss'])
axs[0, 1].plot(history.history['val_loss'])
axs[0, 1].set_title('Model loss')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].legend(['Train', 'Val'], loc='upper left')

# Evaluate the model
scores = model.evaluate(E_test, Y_test)
print("\nEvaluation sur le test data %s: %.2f - %s: %.2f%% " % (
    model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

# Evaluate with the entire dataset and plot
prediction = model.predict_on_batch(Et)
prediction = prediction.reshape(10, 10)
attendues = Y.reshape(10, 10)

# Plot Cartography
axs[1, 0].imshow(attendues, extent=[0, 1, 0, 1])
axs[1, 0].set_title('Cartographie de la fonction attendue')
axs[1, 0].set_xlabel('Entree 1')

axs[1, 1].imshow(prediction, extent=[0, 1, 0, 1])
axs[1, 1].set_title('Cartographie de la fonction predite')
axs[1, 1].set_xlabel('Entree 1')
axs[1, 1].set_ylabel('Entree 2')

# Show the figure
plt.tight_layout()
plt.show()