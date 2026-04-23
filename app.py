from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import json
import logging
from datetime import datetime
import os
import mlflow
import numpy as np

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------
# App Init
# -----------------------------
app = FastAPI(title="House Price Prediction API")



# -----------------------------
# Load Model from MLflow Registry
# -----------------------------
# model = mlflow.pyfunc.load_model("models:/house-price-model/1")


if not MLFLOW_URI:
    raise ValueError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(MLFLOW_URI)

model = None
def get_model():
    global model
    if model is None:
         model = mlflow.pyfunc.load_model("models:/house-price-model@production")
    return model


# -----------------------------
# Input Schema (Validation)
# -----------------------------
class HouseRequest(BaseModel):
    living_area: float = Field(..., gt=0)
    number_of_bedrooms: int = Field(..., gt=0)
    number_of_bathrooms: float = Field(..., gt=0)
    number_of_floors: float = Field(..., gt=0)
    grade_of_the_house: int = Field(..., gt=0)
    house_age: int = Field(..., ge=0)
    area_per_bedroom: float = Field(..., gt=0)

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(request: HouseRequest):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([request.dict()])
        data = data.astype({
        "living_area": "int64",
        "number_of_bedrooms": "int64",
        "number_of_bathrooms": "float64",
        "number_of_floors": "float64",
        "grade_of_the_house": "int64",
        "house_age": "int64",
        "area_per_bedroom": "float64"
        })

        # Predict (log scale)
        prediction_log = get_model().predict(data)[0]

        # Convert back to original price
        prediction = float(np.expm1(prediction_log))

        # Round value
        rounded_price = round(prediction, 2)

        # Format (international)
        formatted_price = f"{rounded_price:,.2f}"

        # Indian format (lakhs)
        price_lakhs = round(prediction / 100000, 2)

        response = {
            "prediction": {
                "value": rounded_price,
                "currency": "INR",
                "formatted": formatted_price,
                "in_lakhs": f"{price_lakhs} Lakhs"
            }
        }
        # -----------------------------
        # Monitoring Log (JSON)
        # -----------------------------
        log_data = {
            "timestamp": str(datetime.utcnow()),
            "input": request.dict(),
            "prediction": rounded_price
        }

        with open("monitoring_log.json", "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # -----------------------------
        # App Logging
        # -----------------------------
        logging.info(f"SUCCESS | Input: {request.dict()} | Prediction: {rounded_price}")

        return response

    except Exception as e:
        logging.error(f"ERROR | Input: {request.dict()} | Error: {str(e)}")
        return {"error": str(e)}   # 👈 temporarily expose error

@app.get("/logs")
def get_logs():
    try:
        with open("monitoring_log.json", "r") as f:
            logs = f.readlines()

        return {
            "logs": logs[-10:]  # last 10 entries
        }

    except Exception as e:
        return {"error": str(e)}