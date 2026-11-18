import os, joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pipeline import validate_input, engineer_features

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ROOT,"artifacts","kiln_model.joblib"))
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
app = FastAPI(title="Ceramic Kiln Energy Prediction API", version="1.0.0")

class KilnBatch(BaseModel):
    pottery_load_kg: float = Field(gt=0)
    target_temperature_c: float = Field(ge=100, le=1500)
    initial_moisture_pct: float = Field(ge=0, le=100)
    planned_duration_hr: float = Field(gt=0)
    insulation_rating: float = Field(gt=0, le=1)
    kiln_volume_m3: float = Field(gt=0)
    ambient_temperature_c: float

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":bundle is not None}

@app.post("/predict")
def predict(batch: KilnBatch):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not trained. Run python src/train.py first.")
    data = batch.model_dump()
    try:
        validate_input(data)
        row = engineer_features(pd.DataFrame([data]))
        prediction = float(bundle["model"].predict(row[bundle["features"]])[0])
        return {"predicted_energy_kwh": round(max(0.0,prediction),2),
                "unit":"kWh","model":bundle["model_name"]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
