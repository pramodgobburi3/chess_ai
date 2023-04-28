import matplotlib.pyplot as plt
import tensorflow as tf

unet_model = tf.keras.models.load_model("unet_model.h5")

# Get the training history
history = unet_model.history.history

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
