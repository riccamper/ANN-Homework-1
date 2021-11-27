#############################
#							#
#       DataLoader.py		#
#							#
#############################

# Import needed libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data loader function (no preprocessing other than the rescaling)
def loadData(training_dir, validation_dir, seed, batch_size):
    # Images are divided into folders, one for each class.
    # If the images are organized in such a way, we can exploit the
    # ImageDataGenerator to read them from disk.

    # Create an instance of ImageDataGenerator with Data Augmentation
    train_data_gen = ImageDataGenerator(rotation_range=30,
                                        height_shift_range=50,
                                        width_shift_range=50,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        rescale=1/255.)

    # Create an instance of ImageDataGenerator with no Data Augmentation
    valid_data_gen = ImageDataGenerator(rescale=1/255.)

    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                   target_size=(256, 256),
                                                   color_mode='rgb',
                                                   classes=None,  # can be set to labels
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=seed)

    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    valid_gen = valid_data_gen.flow_from_directory(directory=validation_dir,
                                                   target_size=(256, 256),
                                                   color_mode='rgb',
                                                   classes=None,  # can be set to labels
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   seed=seed)

    # Show dataset information
    print("Assigned labels training set: ")
    print(train_gen.class_indices)
    print("Target classes training set: ")
    print(train_gen.classes)
    print()

    print("Assigned labels validation set: ")
    print(valid_gen.class_indices)
    print("Target classes validation set: ")
    print(valid_gen.classes)
    print()

    # Return dataset
    return {"train": train_gen, "validation": valid_gen}

#Function for loading input data using a preprocessing function
def loadPreprocessedData(training_dir, validation_dir, seed, batch_size, preprocess_input):
    train_data_gen = ImageDataGenerator(rotation_range=30,
                                            height_shift_range=50,
                                            width_shift_range=50,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            preprocessing_function=preprocess_input) 
    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                        target_size=(256,256),
                                                        color_mode='rgb',
                                                        classes=None, # can be set to labels
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=seed)


    valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    valid_gen = valid_data_gen.flow_from_directory(directory=validation_dir,
                                                target_size=(256,256),
                                                color_mode='rgb',
                                                classes=None, # can be set to labels
                                                class_mode='categorical',
                                                batch_size=batch_size,
                                                shuffle=False,
                                                seed=seed)

    # Show dataset information
    print("Assigned labels training set: ")
    print(train_gen.class_indices)
    print("Target classes training set: ")
    print(train_gen.classes)
    print()

    print("Assigned labels validation set: ")
    print(valid_gen.class_indices)
    print("Target classes validation set: ")
    print(valid_gen.classes)
    print()

    # Return dataset
    return {"train": train_gen, "validation": valid_gen}