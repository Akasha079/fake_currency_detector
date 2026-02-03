# Fake Currency Detector

This project allows for the detection of fake currency using a CNN model deployed via FastAPI.

## Structure

- `app/`: Contains the application source code.
- `model/`: Contains the trained CNN model.
- `data/`: Directory for datasets.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `uvicorn app.main:app --reload`
