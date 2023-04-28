from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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


vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(256, 256, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(10, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

    # serialize the model to disk
print("[INFO] saving object detector model...")
model.save("model", save_format="h5")
# plot the model training history
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
