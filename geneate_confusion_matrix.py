import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Define the test data directory
test_data_dir = "data/squares/validation"

# Get the true labels and image file paths for the test dataset
true_labels, test_image_paths = [], []

class_labels = ['_b', '_k', '_n', '_p', '_q', '_r', 'B', 'f', 'K', 'N', 'P', 'Q', 'R']
predict_labels = ['B', 'K', 'N', 'P', 'Q', 'R', '_b', '_k', '_n', '_p', '_q', '_r', 'f']
img_size = (64, 64)

model_path = "chess_piece_classifier.h5"
model = load_model(model_path)

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

for label in class_labels:
    class_folder = os.path.join(test_data_dir, label)
    image_files = os.listdir(class_folder)

    for img_file in image_files:
        true_labels.append(label)
        test_image_paths.append(os.path.join(class_folder, img_file))

# Encode the true labels using a LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(class_labels)
true_labels_encoded = label_encoder.transform(true_labels)

# Make predictions for the test images
predicted_labels = []
for img_path in test_image_paths:
    predicted_label = predict_chess_piece(img_path)
    predicted_labels.append(predicted_label)

# Encode the predicted labels
predicted_labels_encoded = label_encoder.transform(predicted_labels)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels_encoded, predicted_labels_encoded)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
