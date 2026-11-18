# Ceramic Kiln Energy Consumption – MLOps

Predict electricity consumption (kWh) for a scheduled ceramic firing batch.

## Dataset
- `data/kiln_batches_v1.csv`: 300 synthetic batches representing the baseline kiln.
- `data/kiln_batches_v2.csv`: 300 synthetic batches representing an older kiln. Insulation ratings are degraded and energy usage is increased.
- `data/feature_store_materialized.csv`: engineered features for reuse.

## Features
`thermal_load_index = pottery_load_kg × target_temperature_c × planned_duration_hr / (kiln_volume_m3 × 1000)`

`moisture_removal_load = pottery_load_kg × initial_moisture_pct / 100`

The feature-store file is a reproducible local materialization. In a production deployment, replace it with Feast or another managed feature store.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train, compare and register
```bash
python src/train.py
mlflow ui --backend-store-uri ./mlruns
```
The script compares Linear Regression and Random Forest using MAE, RMSE and R², logs both runs, selects the lowest-MAE model, saves a serving artifact and attempts to register it as `CeramicKilnEnergyModel`. MLflow Model Registry requires a registry-capable backend; for local registry workflows, configure a SQLite backend store and artifact store as described in MLflow documentation.

## Run API
```bash
uvicorn src.api:app --reload
```
Open `http://127.0.0.1:8000/docs`.

Example request:
```json
{
  "pottery_load_kg": 80,
  "target_temperature_c": 1100,
  "initial_moisture_pct": 8,
  "planned_duration_hr": 10,
  "insulation_rating": 0.8,
  "kiln_volume_m3": 1.2,
  "ambient_temperature_c": 25
}
```
Response contains `predicted_energy_kwh`, unit and model name. Predictions are clipped to non-negative values.

## CI/CD
GitHub Actions runs pytest on pushes and pull requests. Tests cover valid and invalid inputs and the non-negative output contract.

## Reproducibility
Synthetic records are generated with a fixed seed (42). Dataset version 2 uses degraded insulation and an energy penalty to model an older kiln. This is a teaching/demo dataset, not measured industrial data.
