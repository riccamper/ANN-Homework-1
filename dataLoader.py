#############################
#							#
# DataLoader.py				#
#							#
#############################

# Import needed libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Data loader function
def loadData(training_dir, validation_split, seed):
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
                                        #fill_mode='reflect',
                                        rescale=1/255.,  # rescale value is multiplied to the image
                                        validation_split=validation_split)

    # Create an instance of ImageDataGenerator with no Data Augmentation
    train_data_gen_no_aug = ImageDataGenerator(rescale=1/255.,
                                               validation_split=validation_split)

    # Create an instance of ImageDataGenerator with no Data Augmentation
    valid_data_gen = ImageDataGenerator(rescale=1/255.,
                                        validation_split=validation_split)

    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                   target_size=(256, 256),
                                                   color_mode='rgb',
                                                   classes=None,  # can be set to labels
                                                   class_mode='categorical',
                                                   batch_size=8,
                                                   shuffle=True,
                                                   seed=seed,
                                                   subset='training')

    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    train_gen_no_aug = train_data_gen_no_aug.flow_from_directory(directory=training_dir,
                                                                 target_size=(
                                                                     256, 256),
                                                                 color_mode='rgb',
                                                                 classes=None,  # can be set to labels
                                                                 class_mode='categorical',
                                                                 batch_size=8,
                                                                 shuffle=True,
                                                                 seed=seed,
                                                                 subset='training')

    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    valid_gen = valid_data_gen.flow_from_directory(directory=training_dir,
                                                   target_size=(256, 256),
                                                   color_mode='rgb',
                                                   classes=None,  # can be set to labels
                                                   class_mode='categorical',
                                                   batch_size=8,
                                                   shuffle=True,
                                                   seed=seed,
                                                   subset='validation')

    # Show dataset information
    print("Assigned labels training set: ")
    print(train_gen.class_indices)
    print("Target classes training set: ")
    print(train_gen.classes)
    print()

    # Return dataset
    return {"train": train_gen, "train_no_aug": train_gen_no_aug, "validation": valid_gen}
