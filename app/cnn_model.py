import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(224, 224, 3)):
    """
    Creates and returns the CNN model architecture.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax') # Assuming 2 classes: Real, Fake
    ])
    return model
