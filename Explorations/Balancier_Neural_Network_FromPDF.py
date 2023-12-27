# Create your first MLP in Keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import timeit
import os

# fix random seed for reproducibility
np.random.seed(7)
# load dataset
dataset = pd.read_csv('venv/outputLabelisedXYZ/labelised_all_axes.csv')

# Use only columns for AccX and AccY as inputs
X = dataset[['AccX [mg]', 'AccY [mg]']].values
# Use only columns for LabelX and LabelY as outputs
Y = dataset[['LabelX', 'LabelY']].values

# create model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='tanh'))  # 2 features as input
model.add(Dense(2, activation='sigmoid'))  # 2 output classes (LabelX and LabelY)

# Optimizer
opt = optimizers.Adam(learning_rate=0.01)

# Compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model
start_time = timeit.default_timer()
history = model.fit(X, Y, validation_split=0.33, epochs=50, batch_size=5, verbose=1)
print("Time elapsed: %.2fs" % (timeit.default_timer() - start_time))

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Save the model if accuracy > 80% and loss < 10%
if scores[1] > 0.8 and scores[0] < 0.1:
    output_folder = 'outputModels'
    os.makedirs(output_folder, exist_ok=True)
    model_filename = f"{output_folder}/Accuracy{scores[1]:.2f}_Loss{scores[0]:.2f}.h5"
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

# Visualization of results
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
