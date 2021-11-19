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
from buildModel import buildModel, buildModelVGG16, buildModelVGG16FT, trainingCallbacks, f1
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
batch_size = 8
train_val_gen = loadData(training_dir, 0.1, seed, batch_size)
train_gen = train_val_gen['train_no_aug']
valid_gen = train_val_gen['validation']

# Model metadata
input_shape = (256, 256, 3)
classes = 14
epochs = 200
folder_name = 'CNN'
model_name = 'CNN'
now = datetime.now().strftime('%b%d_%H-%M-%S')

# Ask for model restoration (Transfer Learning)
restore = input('Do you want to restore a model (Transfer Learning)? Y/N : ')
if restore.upper() == 'Y':
	# Restore model
	model_to_restore = input('Insert the model name: ')
	model = tfk.models.load_model(folder_name + "/" + model_to_restore + "/model", custom_objects={'f1':f1})
	history = np.load(folder_name + "/" + model_to_restore +
						"/history.npy", allow_pickle='TRUE').item()
else:
	# Build model (for data augmentation training)
	model = buildModel(input_shape, classes, tfk, tfkl, seed)
	#model = buildModelVGG16(input_shape, classes, tfk, tfkl, seed)

	# Checkpoint restoration
	#restore = input('Do you want to restore a checkpoint? Y/N : ')
	#if restore.upper() == 'Y':
		# Restore checkpoint
	#	check_to_restore = input('Insert the checkpoint name: ')
	#	model.load_weights(folder_name + "/" + check_to_restore + "/ckpts")

	# Create folders
	exps_dir = os.path.join(folder_name)
	if not os.path.exists(exps_dir):
		os.makedirs(exps_dir)
	exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	ckpt_dir = os.path.join(exp_dir, 'ckpts_ft')
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	# Callbacks
	callbacks = trainingCallbacks(
		ckpt_dir=ckpt_dir, logs=False)

	# Train the model
	history = model.fit(
		x=train_gen,
		epochs=epochs,
		validation_data=valid_gen,
		callbacks=callbacks,
	).history

	# Save best epoch model
	model.save(folder_name + "/" + model_name + '_' + str(now) + '/model')
	np.save(folder_name + "/" + model_name + '_' +
			str(now) + "/history.npy", history)


# Ask for model restoration (Fine Tuning)
restore = input('Do you want to restore a model (Fine Tuning)? Y/N : ')
if restore.upper() == 'Y':
	# Restore model
	model_to_restore = input('Insert the model name: ')
	model = tfk.models.load_model(folder_name + "/" + model_to_restore + "/model_ft", custom_objects={'f1':f1})
	history = np.load(folder_name + "/" + model_to_restore +
						"/history_ft.npy", allow_pickle='TRUE').item()
else:
	# Build model (Fine Tuning)
	model = buildModelVGG16FT(model, tfk)

	# Create folders
	exps_dir = os.path.join(folder_name)
	if not os.path.exists(exps_dir):
		os.makedirs(exps_dir)
	exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	ckpt_dir = os.path.join(exp_dir, 'ckpts')
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	# Callbacks
	callbacks = trainingCallbacks(
		ckpt_dir=ckpt_dir, logs=False)

	# Train the model
	history = model.fit(
		x=train_gen,
		epochs=epochs,
		validation_data=valid_gen,
		callbacks=callbacks,
	).history

	# Save best epoch model
	model.save(folder_name + "/" + model_name + '_' + str(now) + '/model_ft')
	np.save(folder_name + "/" + model_name + '_' +
			str(now) + "/history_ft.npy", history)

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
