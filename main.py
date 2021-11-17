#############################
#							#
# Main.py					#
#							#
#############################

# Import needed libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress warnings
from buildModel import buildModel, buildModelVGG16, trainingCallbacks
from dataLoader import loadData
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
from datetime import datetime
import numpy as np

# Init message
print('')
print(' *** I tre neuroni ***')
print('')


# Test Keras version
tfk = tf.keras
tfkl = tf.keras.layers
print('Keras version: ' + tf.__version__)
print('')

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Dataset folders
dataset_dir = 'dataset'
training_dir = os.path.join(dataset_dir, 'training')
# validation_dir = os.path.join(dataset_dir, 'validation')
# test_dir = os.path.join(dataset_dir, 'test')

# Load dataset
train_val_gen = loadData(training_dir, 0.1, seed)
train_gen = train_val_gen['train']
valid_gen = train_val_gen['validation']

# Model metadata
input_shape = (256, 256, 3)
classes = 14
epochs = 200
model_name = 'CNN'
folder_name = 'CNN'
now = datetime.now().strftime('%b%d_%H-%M-%S')

# Ask for model restoration
restore = input('Do you want to restore a model? Y/N')
if restore.upper() == 'Y':
    model = tfk.models.load_model(folder_name + "/" + model_name + "_best")
    history = np.load(folder_name + "/" + model_name +
                      "_best_history.npy", allow_pickle='TRUE').item()
else:
    # Build model (for data augmentation training)
    # model = buildModel(input_shape, classes, tfk, tfkl, seed)
	model = buildModelVGG16(input_shape, classes, tfk, tfkl, seed)

    # Create folders and callbacks and fit
	callbacks = trainingCallbacks(
        model_name=model_name, folder_name=folder_name, logs=False)

    # Train the model
	history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
    ).history

    # Save best epoch model
	model.save(folder_name + "/" + model_name + '_' + str(now) + "_best")
	np.save(folder_name + "/" + model_name + '_' +
            str(now) + "_best_history.npy", history)

# Plot the training
plt.figure(figsize=(15, 5))
plt.plot(history['loss'], label='Training',
         alpha=.3, color='#ff7f0e', linestyle='--')
plt.plot(history['val_loss'],
         label='Validation', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Categorical Crossentropy')
plt.grid(alpha=.3)
plt.figure(figsize=(15, 5))
plt.plot(history['accuracy'], label='Training',
         alpha=.8, color='#ff7f0e', linestyle='--')
plt.plot(history['val_accuracy'],
         label='Validation', alpha=.8, color='#ff7f0e')
plt.legend(loc='upper left')
plt.title('Accuracy')
plt.grid(alpha=.3)
plt.show()

# Evaluation
model_metrics = model.evaluate(valid_gen, return_dict=True)

print()
print("Test metrics:")
print(model_metrics)
