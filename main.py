#############################
#							#
# Main.py					#
#							#
#############################

# Init message
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import random
import tensorflow as tf
from dataLoader import loadData
from buildModel import buildModel, trainingCallbacks
import os
print('')
print(' *** I tre neuroni ***')
print('')

# Import needed libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings
tf.get_logger().setLevel('ERROR')

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
#validation_dir = os.path.join(dataset_dir, 'validation')
#test_dir = os.path.join(dataset_dir, 'test')

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

# Callback registration
callbacks = trainingCallbacks(
    model_name=model_name, folder_name=folder_name, logs=False)

# Ask for model restoration
restore = input('Do you want to restore a model? Y/N')
if restore.upper() == 'Y':
    model = tfk.models.load_model(folder_name + "/" + model_name + "_best")
else:
    # Build model (for data augmentation training)
    model = buildModel(input_shape, classes, tfk, tfkl, seed)

    # Train the model
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
    ).history

    # Save best epoch model
    model.save(folder_name + "/" + model_name + '_' + str(now) + "_best")

# Evaluation
model_metrics = model.evaluate(valid_gen, return_dict=True)

print()
print("Test metrics:")
print(model_metrics)
