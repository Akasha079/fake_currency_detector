import numpy as np
from tensorflow.keras.models import load_model
from app.preprocess import preprocess_image

MODEL_PATH = "model/currency_cnn.h5"
model = load_model(MODEL_PATH)

LABELS = ["Real", "Fake"]

def predict_currency(image):
    preprocessed = preprocess_image(image)
    preprocessed = np.expand_dims(preprocessed, axis=0)  # Batch dimension
    pred = model.predict(preprocessed, verbose=0)[0]
    label = LABELS[np.argmax(pred)]
    confidence = float(np.max(pred))
    return label, confidence
