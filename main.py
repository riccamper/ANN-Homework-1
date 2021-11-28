#############################
#         MAIN.PY           #
#############################

# Import needed libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress warnings
from buildModel import buildModel, buildModelAlpha, buildModelVGG16TL, buildModelVGG16FT, buildModelInceptionTL, buildModelInceptionFT
from dataLoader import loadData, loadPreprocessedData
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from datetime import datetime
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception

# Init message
print('')
print(' *** I tre neuroni ***')
print('')

# Test Tensorflow version
tfk = tf.keras
tfkl = tf.keras.layers
print('Tensorflow version: ' + tf.__version__)
print('')

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Labels for confusion matrix
labels = {
    0:'Apple', 
    1:'Blueberry', 
    2:'Cherry', 
    3:'Corn', 
    4:'Grape', 
    5:'Orange', 
    6:'Peach', 
    7:'Pepper', 
    8:'Potato', 
    9:'Raspberry',
    10:'Soybean', 
    11:'Squash', 
    12:'Strawberry', 
    13:'Tomato', 
}

# Folder settings
now = datetime.now().strftime('%b%d_%H-%M-%S')
dataset_dir = 'dataset'
model_dir = 'CNN'

# Dataset selection (undersampling or normal samples)
if_undersampled = input('Do you want to use an undersampled dataset? (Y/N): ')
print('')
if if_undersampled.upper() == 'Y': dataset_dir = os.path.join(dataset_dir, 'undersampled')

int_dataset = input('Which premade dataset do you want to use? (1-5): ')
print('')
dataset_dir = os.path.join(dataset_dir, 'set' + str(int_dataset))
training_dir = os.path.join(dataset_dir, 'training')
validation_dir = os.path.join(dataset_dir, 'validation')

print('Using dataset from folder: ' + dataset_dir)
print('')

# Network selection
print('Select the network to be used')
print('1. Simple network with 5 convolutional layers')
print('2. VGG16-like network with less convolutional layers')
print('3. VGG16 (Transfer Learning + Fine tuning)')
print('4. Inception ResNet V2 (Transfer Learning + Fine tuning)')
model_type = input('')
print('')

#####################
#   Simple Network  #
#####################

if model_type == '1':
    # Define model metadata
    input_shape = (256, 256, 3)
    classes = 14
    epochs = 200

    # Load dataset
    batch_size = 32
    train_val_gen = loadData(training_dir, validation_dir, seed, batch_size)
    train_gen = train_val_gen['train']
    valid_gen = train_val_gen['validation']

    # Create model
    model = buildModel(input_shape, classes, tfk, tfkl, seed)

    # Create folders
    exps_dir = os.path.join(model_dir)
    if not os.path.exists(exps_dir): os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, 'CNN_' + str(now))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    # Train the model
    history = model.fit(
        x = train_gen,
        epochs = epochs,
        validation_data = valid_gen,
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)]
    ).history

    # Save best epoch model
    print()
    model.save(model_dir + "/" + str(now) + '/model')
    np.save(model_dir + "/" + str(now) + "/history.npy", history)
    print()

    # Plot the training history
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
    predictions = model.predict(valid_gen)

    # Compute the confusion matrix
    cm = confusion_matrix(valid_gen.classes, np.argmax(predictions, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(valid_gen.classes, np.argmax(predictions, axis=-1))
    precision = precision_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    print()
    print('Validation Metrics:')
    print('Accuracy:',accuracy.round(4))
    print('Precision:',precision.round(4))
    print('Recall:',recall.round(4))
    print('F1:',f1.round(4))

    #Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm.T, xticklabels=list(labels.values()), yticklabels=list(labels.values()))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()

#########################
#   VGG-like Network    #
#########################

elif model_type == '2':
    # Define model metadata
    input_shape = (256, 256, 3)
    classes = 14
    epochs = 200

    # Load dataset
    batch_size = 32
    train_val_gen = loadData(training_dir, validation_dir, seed, batch_size)
    train_gen = train_val_gen['train']
    valid_gen = train_val_gen['validation']

    # Create model
    model = buildModelAlpha(input_shape, classes, tfk, tfkl, seed)

    # Create folders
    exps_dir = os.path.join(model_dir)
    if not os.path.exists(exps_dir): os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, 'CNN_' + str(now))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    # Train the model
    history = model.fit(
        x = train_gen,
        epochs = epochs,
        validation_data = valid_gen,
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)]
    ).history

    # Save best epoch model
    print()
    model.save(model_dir + "/" + str(now) + '/model')
    np.save(model_dir + "/" + str(now) + "/history.npy", history)
    print()

    # Plot the training history
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
    predictions = model.predict(valid_gen)

    # Compute the confusion matrix
    cm = confusion_matrix(valid_gen.classes, np.argmax(predictions, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(valid_gen.classes, np.argmax(predictions, axis=-1))
    precision = precision_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    print()
    print('Validation Metrics:')
    print('Accuracy:',accuracy.round(4))
    print('Precision:',precision.round(4))
    print('Recall:',recall.round(4))
    print('F1:',f1.round(4))

    #Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm.T, xticklabels=list(labels.values()), yticklabels=list(labels.values()))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()


#######################################################
#   VGG16 Network (Transfer Learning & Fine Tuning)   #
#######################################################

elif model_type == '3':
    # Define model metadata
    input_shape = (256, 256, 3)
    classes = 14
    tl_epochs = 200
    ft_epochs = 200

    # Load dataset
    batch_size = 32
    train_val_gen = loadPreprocessedData(training_dir, validation_dir, seed, batch_size, preprocess_vgg16)
    train_gen = train_val_gen['train']
    valid_gen = train_val_gen['validation']

    ### TRANSFER LEARNING ###

    # Create model
    model = buildModelVGG16TL(input_shape, classes, tfk, tfkl, seed)

    # Create folders
    exps_dir = os.path.join(model_dir)
    if not os.path.exists(exps_dir): os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, 'CNN_' + str(now))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    # Train the model
    history = model.fit(
        x = train_gen,
        epochs = tl_epochs,
        validation_data = valid_gen,
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)]
    ).history

    # Save best epoch model
    print()
    model.save(model_dir + "/" + str(now) + '/model_tl')
    np.save(model_dir + "/" + str(now) + "/history_tl.npy", history)
    print()

    ### FINE TUNING ###

    # Create model
    model = buildModelVGG16FT(model, tfk)

    # Create folders
    exps_dir = os.path.join(model_dir)
    if not os.path.exists(exps_dir): os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, 'CNN_' + str(now))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    # Train the model
    history_ft = model.fit(
        x = train_gen,
        epochs = ft_epochs,
        validation_data = valid_gen,
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)]
    ).history

    # Save best epoch model
    print()
    model.save(model_dir + "/" + str(now) + '/model_ft')
    np.save(model_dir + "/" + str(now) + "/history_ft.npy", history_ft)
    print()

    # Plot the training history
    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], label='Training TL',
            alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'],
            label='Validation TL', alpha=.8, color='#ff7f0e')
    plt.plot(history_ft['loss'], label='Training FT',
            alpha=.3, color='#8fce00', linestyle='--')
    plt.plot(history_ft['val_loss'],
            label='Validation FT', alpha=.8, color='#8fce00')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)
    plt.figure(figsize=(15, 5))
    plt.plot(history['accuracy'], label='Training TL',
            alpha=.8, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'],
            label='Validation TL', alpha=.8, color='#ff7f0e')
    plt.plot(history_ft['accuracy'], label='Training FT',
            alpha=.8, color='#8fce00', linestyle='--')
    plt.plot(history_ft['val_accuracy'],
            label='Validation FT', alpha=.8, color='#8fce00')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)
    plt.show()

    # Evaluation
    predictions = model.predict(valid_gen)

    # Compute the confusion matrix
    cm = confusion_matrix(valid_gen.classes, np.argmax(predictions, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(valid_gen.classes, np.argmax(predictions, axis=-1))
    precision = precision_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    print()
    print('Validation Metrics:')
    print('Accuracy:',accuracy.round(4))
    print('Precision:',precision.round(4))
    print('Recall:',recall.round(4))
    print('F1:',f1.round(4))

    #Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm.T, xticklabels=list(labels.values()), yticklabels=list(labels.values()))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()

#############################################################
#   Inception ResNet V2 (Transfer Learning + Fine Tuning)   #
#############################################################

elif model_type == '4':
    # Define model metadata
    input_shape = (256, 256, 3)
    classes = 14
    tl_epochs = 60
    ft_epochs = 50

    # Load dataset
    batch_size = 32
    train_val_gen = loadPreprocessedData(training_dir, validation_dir, seed, batch_size, preprocess_inception)
    train_gen = train_val_gen['train']
    valid_gen = train_val_gen['validation']

    ### TRANSFER LEARNING ###

    # Create model
    model = buildModelInceptionTL(input_shape, classes, tfk, tfkl, seed)

    # Create folders
    exps_dir = os.path.join(model_dir)
    if not os.path.exists(exps_dir): os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, 'CNN_' + str(now))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    # Train the model
    history = model.fit(
        x = train_gen,
        epochs = tl_epochs,
        validation_data = valid_gen,
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
    ).history

    # Save best epoch model
    print()
    model.save(model_dir + "/" + str(now) + '/model_tl')
    np.save(model_dir + "/" + str(now) + "/history_tl.npy", history)
    print()

    ### FINE TUNING ###

    # Create model
    model = buildModelInceptionFT(model, tfk)

    # Create folders
    exps_dir = os.path.join(model_dir)
    if not os.path.exists(exps_dir): os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, 'CNN_' + str(now))
    if not os.path.exists(exp_dir): os.makedirs(exp_dir)

    # Train the model
    history_ft = model.fit(
        x = train_gen,
        epochs = ft_epochs,
        validation_data = valid_gen,
        callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
    ).history

    # Save best epoch model
    print()
    model.save(model_dir + "/" + str(now) + '/model_ft')
    np.save(model_dir + "/" + str(now) + "/history_ft.npy", history_ft)
    print()

    # Plot the training history
    plt.figure(figsize=(15, 5))
    plt.plot(history['loss'], label='Training TL',
            alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'],
            label='Validation TL', alpha=.8, color='#ff7f0e')
    plt.plot(history_ft['loss'], label='Training FT',
            alpha=.3, color='#8fce00', linestyle='--')
    plt.plot(history_ft['val_loss'],
            label='Validation FT', alpha=.8, color='#8fce00')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)
    plt.figure(figsize=(15, 5))
    plt.plot(history['accuracy'], label='Training TL',
            alpha=.8, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'],
            label='Validation TL', alpha=.8, color='#ff7f0e')
    plt.plot(history_ft['accuracy'], label='Training FT',
            alpha=.8, color='#8fce00', linestyle='--')
    plt.plot(history_ft['val_accuracy'],
            label='Validation FT', alpha=.8, color='#8fce00')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)
    plt.show()

    # Evaluation
    predictions = model.predict(valid_gen)

    # Compute the confusion matrix
    cm = confusion_matrix(valid_gen.classes, np.argmax(predictions, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(valid_gen.classes, np.argmax(predictions, axis=-1))
    precision = precision_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(valid_gen.classes, np.argmax(predictions, axis=-1), average='macro')
    print()
    print('Validation Metrics:')
    print('Accuracy:',accuracy.round(4))
    print('Precision:',precision.round(4))
    print('Recall:',recall.round(4))
    print('F1:',f1.round(4))

    #Plot the confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm.T, xticklabels=list(labels.values()), yticklabels=list(labels.values()))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()
	