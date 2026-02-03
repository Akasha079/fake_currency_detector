import cv2
import numpy as np

def preprocess_image(image, target_size=(128,128)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize
    resized = cv2.resize(gray, target_size)
    # Normalize
    normalized = resized / 255.0
    # Expand dims for CNN
    return np.expand_dims(normalized, axis=-1)
