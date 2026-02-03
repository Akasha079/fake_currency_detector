from fastapi import FastAPI, UploadFile, File
from app.predict import predict_currency
from app.schemas import CurrencyResponse
import cv2
import numpy as np

app = FastAPI(title="Fake Currency Detector")

@app.post("/detect", response_model=CurrencyResponse)
async def detect_currency(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    label, confidence = predict_currency(image)
    return {"label": label, "confidence": confidence}
