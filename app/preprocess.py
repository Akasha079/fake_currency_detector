from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes: bytes, target_size=(224, 224)):
    """
    Preprocess the image bytes into a format suitable for the model.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array
