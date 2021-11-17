#############################
#							#
# Main.py					#
#							#
#############################

# Init message
print('')
print(' *** I tre neuroni ***')
print('')

# Import needed libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings
from buildModel import buildModel, trainingCallbacks
from dataLoader import loadData
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np 
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from PIL import Image

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
train_gen = train_val_gen['train_no_aug']
valid_gen = train_val_gen['validation']

# Model metadata
input_shape = (256, 256, 3)
classes = 14
epochs = 200
model_name = 'CNN'
folder_name = 'CNN'

# Build model (for data augmentation training)
model = buildModel(input_shape, classes, tfk, tfkl, seed)

# Create folders and callbacks and fit
callbacks = trainingCallbacks(model_name=model_name, folder_name=folder_name, logs=True)

# Train the model
history = model.fit(
    x = train_gen,
    epochs = epochs,
    validation_data = valid_gen,
    callbacks = callbacks,
).history

# Save best epoch model
model.save(folder_name + "/" + model_name + "_best")

# Evaluation
#model_aug_test_metrics = model.evaluate(test_gen, return_dict=True)

#print()
#print("Test metrics with data augmentation")
#print(model_aug_test_metrics)