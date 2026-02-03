import tensorflow as tf
import numpy as np
from app.preprocess import preprocess_image

class Predictor:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Fake', 'Real'] # Update based on actual training

    def predict(self, image_bytes: bytes):
        processed_image = preprocess_image(image_bytes)
        predictions = self.model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            "prediction": self.class_names[predicted_class_index],
            "confidence": confidence
        }

# Singleton instance to be used by main app
# predictor = Predictor("model/currency_cnn.h5") 
