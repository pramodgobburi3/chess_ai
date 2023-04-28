import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set the dataset directory and other parameters
base_dir = "data/squares"
train_dir = os.path.join(base_dir, "training")
val_dir = os.path.join(base_dir, "validation")
img_size = (64, 64)
batch_size = 16
num_classes = 13

# Create a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print(train_generator.class_indices)
print(val_generator.class_indices)

# Train the model
# epochs = 100
# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=val_generator
# )

# # Save the model
# model.save("chess_piece_classifier.h5")

# history = history.history

# # Plot the training and validation loss
# plt.figure()
# plt.plot(history['loss'], label='Training Loss')
# plt.plot(history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.title('Loss vs. Epochs')
# plt.savefig('piece_loss_graph.png')

# # Assuming you have an accuracy metric in your model, plot the training and validation accuracy
# if 'accuracy' in history:
#     plt.figure()
#     plt.plot(history['accuracy'], label='Training Accuracy')
#     plt.plot(history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Accuracy vs. Epochs')
#     plt.savefig('piece_accuracy_graph.png')
