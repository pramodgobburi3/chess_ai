import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from unet_model import unet_model
import matplotlib.pyplot as plt

BASEDIR = os.getcwd()

image_path = os.path.join(BASEDIR, "data/board_extraction/grayscale_images")
mask_path = os.path.join(BASEDIR, "data/board_extraction/masks")

# Load the list of images and masks filepaths
image_files = os.listdir(image_path)
image_files = [os.path.join(image_path, f) for f in image_files] 
mask_files = os.listdir(mask_path)
mask_files = [os.path.join(mask_path, f) for f in mask_files] 

# Split data into training and validation
img_train, img_val, mask_train, mask_val = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)

# Define a custom data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_files, mask_files, batch_size=8, img_size=(256, 256), is_training=True):
        self.img_files = img_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.img_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img_files = self.img_files[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_mask_files = self.mask_files[idx*self.batch_size:(idx+1)*self.batch_size]
        
        X, Y = self.__data_generation(batch_img_files, batch_mask_files)

        return X, Y

    def on_epoch_end(self):
        if self.is_training:
            zipped = list(zip(self.img_files, self.mask_files))
            np.random.shuffle(zipped)
            self.img_files, self.mask_files = zip(*zipped)

    def __data_generation(self, batch_img_files, batch_mask_files):
        X = np.empty((self.batch_size, *self.img_size, 1))
        Y = np.empty((self.batch_size, *self.img_size, 1))

        for i, (img_file, mask_file) in enumerate(zip(batch_img_files, batch_mask_files)):
            # Load and preprocess the image
            img = imread(img_file, as_gray=True)
            img = np.expand_dims(img, axis=-1)
            img = img / 255.  # normalize image from 0-255 to 0-1

            # Load and preprocess the mask
            mask = imread(mask_file, as_gray=True)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask / 255.  # normalize mask from 0-255 to 0-1
            mask = (mask > 0.5).astype(float)  # binarize the mask

            X[i,] = img
            Y[i,] = mask

        return X, Y

# Create data generators for training and validation
train_generator = DataGenerator(img_train, mask_train, batch_size=16, img_size=(256, 256), is_training=True)
val_generator = DataGenerator(img_val, mask_val, batch_size=16, img_size=(256, 256), is_training=False)

# Set some training parameters
epochs = 100
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

# Train the model
history = unet_model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_generator, validation_steps=validation_steps)
unet_model.save("unet_model2.h5")

history = history.history

# Plot the training and validation loss
plt.figure()
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Loss vs. Epochs')
plt.savefig('loss_graph.png')

# Assuming you have an accuracy metric in your model, plot the training and validation accuracy
if 'accuracy' in history:
    plt.figure()
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy vs. Epochs')
    plt.savefig('accuracy_graph.png')
