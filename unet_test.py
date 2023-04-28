import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASEDIR = os.getcwd()

image_path = os.path.join(BASEDIR, "data/board_extraction/grayscale_images")

# Load the pre-trained U-Net model
unet_model = tf.keras.models.load_model("unet_model.h5")

image_files = os.listdir(image_path)
image_files = [os.path.join(image_path, f) for f in image_files] 

# Load and preprocess the input image
input_image_path = image_files[0]
input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (256, 256))
input_image_normalized = input_image / 255.0
input_array = np.expand_dims(input_image_normalized, axis=(0, -1))

# Make prediction using the U-Net model
predicted_mask = unet_model.predict(input_array)

# Display the predicted mask
predicted_mask = np.squeeze(predicted_mask, axis=(0, -1))
plt.imshow(predicted_mask, cmap="gray")
plt.axis("off")
plt.show()
