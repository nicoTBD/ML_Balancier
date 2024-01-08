# Create your first MLP in Keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import timeit
import os

# fix random seed for reproducibility
seed = np.random.seed(7)

# load dataset
dataset = pd.read_csv('outputLabelisedXYZ/labelised_all_axes.csv')

# Use only columns for AccX and AccY as inputs
X = dataset[['AccX [mg]']].values# X = dataset[['AccX [mg]', 'AccY [mg]']].values
# Use only columns for LabelX and LabelY as outputs
Y = dataset[['LabelX']].values# Y = dataset[['LabelX', 'LabelY']].values

# 70-15-15
# Division en ensemble d'entraînement et de test (70% / 30%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# Division de l'ensemble de test en ensembles de validation et de test (50% / 50%)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=seed)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

"""
# create model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='tanh'))  # 2 features as input
model.add(Dense(2, activation='sigmoid'))  # 2 output classes (LabelX and LabelY)

# Optimizer
opt = optimizers.Adam(learning_rate=0.01)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
"""
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu', kernel_regularizer=l2(0.03)))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.03)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.03)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Hyperparameters
nb_epochs = 70
batch_size = 30
learning_rate = 0.01
decay_rate = learning_rate/nb_epochs
momentum = 0.8

# Optimizer
opt = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=False)
#opt = optimizers.Adam(learning_rate=learning_rate)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse', 'acc'])

# Fit the model
start_time = timeit.default_timer()
history = model.fit(X_train_scaled, Y_train, validation_data=(X_val_scaled, Y_val), epochs=nb_epochs, batch_size=batch_size, verbose=1)
print("Time elapsed: %.2fs" % (timeit.default_timer() - start_time))

# evaluate the model
scores = model.evaluate(X_test_scaled, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))

# Save the model if loss < 70% and mse < 30%
if scores[0] < 0.7 and scores[1] < 0.3:
    output_folder = 'outputModels'
    os.makedirs(output_folder, exist_ok=True)
    model_filename = f"{output_folder}/Accuracy{scores[2]:.2f}_Loss{scores[1]:.2f}.h5"

    """
    # Save the model in HDF5 format
    h5_model_filename = f"{output_folder}/modelOnly_Acc{scores[2]:.2f}_Loss{scores[1]:.2f}_MSE{scores[0]:.2f}.h5"
    model.save(h5_model_filename)
    print(f"Model saved in HDF5 format to {h5_model_filename}")
    """

    # Save the weights separately in HDF5 format
    weights_filename = f"{output_folder}/modelWeights_Acc{scores[2]:.2f}_Loss{scores[1]:.2f}_MSE{scores[0]:.2f}.h5"
    model.save_weights(weights_filename)
    print(f"Model Weights saved in HDF5 format to {weights_filename}")

    # Save the model in JSON format
    json_model_filename = f"{output_folder}/modelJSON_Acc{scores[2]:.2f}_Loss{scores[1]:.2f}_MSE{scores[0]:.2f}.json"
    model_json = model.to_json()
    with open(json_model_filename, "w") as json_file:
        json_file.write(model_json)
    print(f"Model saved in JSON format to {json_model_filename}")

# Visualization of results
plt.figure(figsize=(18, 4))

# Plot MSE
plt.subplot(1, 3, 1)
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['acc'])  # Change 'accuracy' to 'acc'
plt.plot(history.history['val_acc'])  # Change 'val_accuracy' to 'val_acc'
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()