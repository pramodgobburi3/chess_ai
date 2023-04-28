import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json

BASEDIR = os.getcwd()
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32

print("loading the dataset")

data_path = os.path.join(BASEDIR, "data.csv")
image_path = os.path.join(BASEDIR, "data/board_extraction/grayscale_images")
mask_path = os.path.join(BASEDIR, "data/board_extraction/masks")

rows = open(data_path).read().strip().split("\n")

data = []
targets = []
filenames = []

for row in rows[1:]:
    row = row.split(";")
    (filename, coordinates) = row
    coordinates = json.loads(coordinates)
    imagePath = os.path.join(image_path, filename)
    image = cv2.imread(imagePath)
    image = load_img(imagePath, target_size=(256,256))
    image = img_to_array(image)
    
    if len(coordinates) == 5:
        el = []
        for c in coordinates:
            (x,y) = c
            el.append(x)
            el.append(y)
        targets.append(el)
        data.append(image)
        filenames.append(filename)


data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open("test_filenames.txt", "w")
f.write("\n".join(testFilenames))
f.close()

base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 1], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16                 
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    pix2pix.upsample(32, 3),   # 64x64 -> 128x128
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #128x128 -> 256x256

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




