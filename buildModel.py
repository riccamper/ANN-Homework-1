#############################
#							#
# BuildModel.py				#
#							#
#############################

# Import needed libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings
from datetime import datetime
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Model builder function
def buildModel(input_shape, classes, tfk, tfkl, seed):
    # (Conv + ReLU + MaxPool) x 5 + FC x 2

    # Input layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # First convolutional layer + activation
    conv1 = tfkl.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed)
    )(input_layer)
    # First pooling layer
    pool1 = tfkl.MaxPooling2D(
        pool_size=(2, 2)
    )(conv1)

    # Second convolutional layer + activation
    conv2 = tfkl.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed)
    )(pool1)
    # Second pooling layer
    pool2 = tfkl.MaxPooling2D(
        pool_size=(2, 2)
    )(conv2)

    # Third convolutional layer + activation
    conv3 = tfkl.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed)
    )(pool2)
    # Third pooling layer
    pool3 = tfkl.MaxPooling2D(
        pool_size=(2, 2)
    )(conv3)

    # 4th convolutional layer + activation
    conv4 = tfkl.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed)
    )(pool3)
    # 4th pooling layer
    pool4 = tfkl.MaxPooling2D(
        pool_size=(2, 2)
    )(conv4)

    # 5th convolutional layer + activation
    conv5 = tfkl.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed)
    )(pool4)
    # 5th pooling layer
    pool5 = tfkl.MaxPooling2D(
        pool_size=(2, 2)
    )(conv5)

    # Flattening (flatten + dropout) layer
    flattening_layer = tfkl.Flatten(name='Flatten')(pool5)
    flattening_layer = tfkl.Dropout(0.3, seed=seed)(flattening_layer)
    # Classification layer (dense + dropout) + activation (relu)
    classifier_layer = tfkl.Dense(units=512, name='Classifier', kernel_initializer=tfk.initializers.GlorotUniform(
        seed), activation='relu')(flattening_layer)
    classifier_layer = tfkl.Dropout(0.3, seed=seed)(classifier_layer)
    # Output layer (Dense) + activation (softmax)
    output_layer = tfkl.Dense(units=classes, activation='softmax', kernel_initializer=tfk.initializers.GlorotUniform(
        seed), name='Output')(classifier_layer)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                  optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model


# Callbacks function for training (callbacks, checkpointing, early stopping)
def trainingCallbacks(model_name):

	# Create folders
    exps_dir = os.path.join('data_augmentation_experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    callbacks = []

    # Model checkpoint
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'),
                                                       save_weights_only=False,  # True to save only weights
                                                       save_best_only=False)  # True to save only the best epoch
    callbacks.append(ckpt_callback)

    # Visualize Learning on Tensorboard
    tb_dir = os.path.join(exp_dir, 'tb_logs')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # By default shows losses and metrics for both training and validation
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                                 profile_batch=0,
                                                 histogram_freq=1)  # if > 0 (epochs) shows weights histograms
    callbacks.append(tb_callback)

    # Early Stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks.append(es_callback)

	# Return callbacks
    return callbacks
