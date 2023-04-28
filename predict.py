from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import cv2
import os

BASEDIR = os.getcwd()
image_path = os.path.join(BASEDIR, "data/board_extraction/grayscale_images")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image filenames")
args = vars(ap.parse_args())

filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [os.path.join(image_path, args["input"])]

print("[INFO] loading object detector...")
model = load_model("model.h5")

def draw_contours(pts):
    mod_pts = []
    for [x,y] in pts:
        mod_pts.append([x *  265, y * 265])

    pts = np.asarray(mod_pts, dtype='int32')

    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2

    polylines = cv2.polylines(image, [pts], True, color, thickness)

    cv2.imshow('polylines', polylines)

    

for imagePath in imagePaths:
    # load the input image (in Keras format) from disk and preprocess
    # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(imagePath, target_size=(256, 256))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]
    preds = np.array_split(preds, 5)
    print(preds)

    image = cv2.imread(imagePath)
    draw_contours(preds)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




