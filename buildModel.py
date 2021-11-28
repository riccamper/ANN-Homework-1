#############################
#       BUILDMODEL.PY       #
#############################

# Base
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings

# Import needed libraries
from keras import backend as K

######################################
#   PARAMETER EVALUATION FUNCTIONS   #
######################################

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision2 = precision(y_true, y_pred)
    recall2 = recall(y_true, y_pred)
    return 2*((precision2*recall2)/(precision2+recall2+K.epsilon()))

################################
#   MODEL CREATION FUNCTIONS   #
################################

#Model builder function (Simple Deep Network)
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
                  optimizer=tfk.optimizers.Adam(), metrics=['accuracy', f1])
                  # optimizer=tfk.optimizers.Adam(), metrics=['accuracy', f1, precision, recall])
    print()
    print('5-layer Network:')
    model.summary()

    # Return the model
    return model

#Model builder function (VGG16-like network)
def buildModelAlpha(input_shape, classes, tfk, tfkl, seed):
    
    model = tfk.Sequential(name = 'Model', layers = [
        tfkl.InputLayer(input_shape=input_shape, name='Input'),

        tfkl.Conv2D(
            filters = 16,
            kernel_size = 5,
            padding = 'same',
            activation = 'relu',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)
        ),
        tfkl.BatchNormalization(),
        tfkl.MaxPooling2D( pool_size = 2 ),

        tfkl.Conv2D(
            filters = 32,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)
        ),
        tfkl.BatchNormalization(),
        tfkl.MaxPooling2D( pool_size = 2 ),

        tfkl.Conv2D(
            filters = 64,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)
        ),
        tfkl.Conv2D(
            filters = 64,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)
        ),
        tfkl.BatchNormalization(),
        tfkl.MaxPooling2D( pool_size = 2 ),

        tfkl.Conv2D(
            filters = 128,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)
        ),
        tfkl.Conv2D(
            filters = 128,
            kernel_size = 3,
            padding = 'same',
            activation = 'relu',
            kernel_initializer=tfk.initializers.GlorotUniform(seed)
        ),
        tfkl.BatchNormalization(),
        tfkl.MaxPooling2D( pool_size = 2 ),

        tfkl.Flatten(),
        tfkl.Dropout( rate = 0.3, seed = seed ),
        tfkl.Dense(
            units = 256,
            name='FC1', 
            kernel_initializer=tfk.initializers.GlorotUniform(seed), 
            activation='relu'
        ),
        tfkl.Dropout( rate = 0.3, seed = seed ),
        tfkl.Dense(
            units = 64,
            name='FC2', 
            kernel_initializer=tfk.initializers.GlorotUniform(seed), 
            activation='relu'
        ),
        tfkl.Dropout( rate = 0.1, seed = seed ),
        tfkl.Dense(
            units=classes, 
            activation='softmax', 
            kernel_initializer=tfk.initializers.GlorotUniform(seed), 
            name='Output'
        )
    ])

    model.compile(
        loss=tfk.losses.CategoricalCrossentropy(),
        optimizer=tfk.optimizers.Adam(learning_rate = 0.001, epsilon = 1e-8), 
        metrics=['accuracy', f1]
    )

    print()
    print('VGG-like Network:')
    model.summary()

    return model

# Model builder function (VGG16 Transfer Learning)
def buildModelVGG16TL(input_shape, classes, tfk, tfkl, seed):
    # VGG16

    # Supernet
    supernet = tfk.applications.VGG16(
        include_top=False,
        weights="imagenet",
        #input_shape=input_shape
		input_shape=(64, 64, 3)
	)

    print()
    print('VGG16 Supernet:')
    supernet.summary()

    # Use the supernet as feature extractor
    supernet.trainable = False

    inputs = tfk.Input(shape=input_shape, name='Input')
    x = tfkl.Resizing(64, 64, interpolation="bicubic")(inputs)
    x = supernet(x)
    x = tfkl.Flatten(name='Flattening')(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    x = tfkl.Dense(
        256,
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    outputs = tfkl.Dense(
        classes,
        activation='softmax',
        kernel_initializer=tfk.initializers.GlorotUniform(seed))(x)

    # Connect input and output through the Model class
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')

    # Compile the model
    tl_model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                     optimizer=tfk.optimizers.Adam(), metrics=['accuracy', f1])
    print()
    print('VGG16 Transfer Learning:')
    tl_model.summary()

    # Return the model
    return tl_model

# Model builder function (VGG16 Fine Tuning)
def buildModelVGG16FT(model, tfk):
    # VGG16 Fine Tuning

	# Set all VGG layers to True
	model.get_layer('vgg16').trainable = True

	# Freeze first N layers, e.g., until 14th
	for i, layer in enumerate(model.get_layer('vgg16').layers[:14]):
		layer.trainable = False

	# Compile the model
	model.compile(loss=tfk.losses.CategoricalCrossentropy(),
	              optimizer=tfk.optimizers.Adam(1e-4), metrics=['accuracy', f1])
	print()
	print('VGG16 Fine Tuning:')
	model.summary()

    # Return the model
	return model

# Model builder function (Inception ResNet V2 Transfer Learning)
def buildModelInceptionTL(input_shape, classes, tfk, tfkl, seed):
    supernet = tfk.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    print()
    print('Inception ResNet V2 Supernet:')
    supernet.summary()

    supernet.trainable = False

    inputs = tfk.Input(shape=input_shape)
    x = supernet(inputs)
    x = tfkl.Flatten(name='Flattening')(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    x = tfkl.Dense(
        512, 
        activation='relu',
        kernel_initializer = tfk.initializers.GlorotUniform(seed))(x)
    x = tfkl.Dropout(0.3, seed=seed)(x)
    outputs = tfkl.Dense(
        classes, 
        activation='softmax',
        kernel_initializer = tfk.initializers.GlorotUniform(seed))(x)


    # Connect input and output through the Model class
    tl_model = tfk.Model(inputs=inputs, outputs=outputs, name='model')

    # Compile the model
    tl_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics=['accuracy',f1])
    print()
    print('Inception ResNet V2 Transfer Learning:')
    tl_model.summary()

    return tl_model

# Model builder function (Inception ResNet V2 Fine Tuning)
def buildModelInceptionFT(ft_model, tfk):
    ft_model.get_layer('inception_resnet_v2').trainable = True

    for layer in ft_model.get_layer('inception_resnet_v2').layers[:617] :
        layer.trainable=False

    ft_model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(1e-4), metrics=['accuracy',f1])

    print()
    print('Inception ResNet V2 Fine Tuning:')
    ft_model.summary()

    return ft_model