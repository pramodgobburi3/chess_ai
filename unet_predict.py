import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def postprocess_mask(mask):
    mask = np.squeeze(mask)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def draw_mask_on_image(src_image, mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    result = cv2.cvtColor(src_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

BASEDIR = os.getcwd()

image_path = os.path.join(BASEDIR, "data/board_extraction/grayscale_images")

image_files = os.listdir(image_path)
image_files = [os.path.join(image_path, f) for f in image_files] 

image_file = image_files[5]

# Load the model
model = tf.keras.models.load_model("unet_model.h5")

# Preprocess the input image
input_image = preprocess_image(image_file)

# Predict the mask
predicted_mask = model.predict(input_image)

# Post-process the mask
mask = postprocess_mask(predicted_mask)

# Load the original image as grayscale
source_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
source_image = cv2.resize(source_image, (256, 256), interpolation=cv2.INTER_AREA)

# Draw the mask outline on the original grayscale image
result_image = draw_mask_on_image(source_image, mask)

# Save the result image
# cv2.imwrite("path/to/save/result/image.jpg", result_image)

cv2.imshow('prediction', result_image)

cv2.waitKey(0)

cv2.destroyAllWindows()


