from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd

# -----------------------------
# App Init
# -----------------------------
app = FastAPI(title="House Price Prediction API")

# -----------------------------
# Load Model from MLflow Registry
# -----------------------------
model = mlflow.pyfunc.load_model("models:/house-price-model/1")

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

        # Predict (log scale)
        prediction_log = model.predict(data)[0]

        # Convert back to original price
        import numpy as np
        prediction = float(np.expm1(prediction_log))

        # Round value
        rounded_price = round(prediction, 2)

        # Format (international)
        formatted_price = f"{rounded_price:,.2f}"

        # Indian format (lakhs)
        price_lakhs = round(prediction / 100000, 2)

        return {
            "predicted_price": prediction,
            "formatted_price": f"₹{formatted_price}",
            "price_in_lakhs": f"{price_lakhs} Lakhs"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))