import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Global model variable
model = None
labels = None

def load_inference_model(model_path="model/mobilenet_model.h5"):
    global model
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def predict_currency(img_array):
    if model is None:
        # Try loading default if not loaded
        load_inference_model()
        if model is None:
            return "Model Error", 0.0

    # Preprocess
    # Resize to 224x224 as expected by MobileNet/ResNet
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Predict
    preds = model.predict(img_batch)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    # Mapping based on typical dataset structure or can be passed dynamically
    # For now, using a standard mapping, but in production, save class_indices.json during training
    # Assuming the classes from the Kaggle dataset example
    class_names = [
        '10', '100', '20', '200', '2000', '50', '500'
    ]
    
    # Fallback to simple Real/Fake if only 2 classes
    if preds.shape[1] == 2:
        class_names = ['Fake', 'Real']

    if class_idx < len(class_names):
        predicted_label = class_names[class_idx]
    else:
        predicted_label = f"Class {class_idx}"

    return predicted_label, confidence
