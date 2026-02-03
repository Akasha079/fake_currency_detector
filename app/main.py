from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.predict import predict_currency
from app.schemas import CurrencyResponse
import cv2
import numpy as np

app = FastAPI(title="Fake Currency Detector")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/detect", response_model=CurrencyResponse)
async def detect_currency(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    label, confidence = predict_currency(image)
    return {"label": label, "confidence": confidence}
