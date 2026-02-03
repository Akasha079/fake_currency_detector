from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import PredictionResponse
# from app.predict import predictor # Uncomment when model is available

app = FastAPI(title="Fake Currency Detector API")

@app.get("/")
async def root():
    return {"message": "Welcome to the Fake Currency Detector API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_currency(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    
    # Placeholder logic until model is present
    # result = predictor.predict(contents)
    
    # Mock response
    result = {
        "prediction": "Real",
        "confidence": 0.98
    }
    
    return {
        "filename": file.filename,
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    }
