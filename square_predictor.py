import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model (replace with the correct fold model if using k-fold)
model_path = "chess_piece_classifier.h5"
model = load_model(model_path)

# Define the class labels
predict_labels = ['B', 'K', 'N', 'P', 'Q', 'R', '_b', '_k', '_n', '_p', '_q', '_r', 'f']

img_size = (64, 64)

def predict_chess_piece(img_path):
    # Preprocess the input image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions using the model
    predictions = model.predict(img_array)
    print(predictions[0])
    predicted_class_idx = np.argmax(predictions[0])
    print(predicted_class_idx)

    # Decode the predictions to obtain the class label
    predicted_label = predict_labels[predicted_class_idx]

    return predicted_label

BASEDIR = os.getcwd()

image_path = os.path.join(BASEDIR, "data/squares/validation/B")

image_files = os.listdir(image_path)
image_files = [os.path.join(image_path, f) for f in image_files] 

img_path = image_files[1]
predicted_label = predict_chess_piece(img_path)
print(f"Predicted chess piece: {predicted_label}")
