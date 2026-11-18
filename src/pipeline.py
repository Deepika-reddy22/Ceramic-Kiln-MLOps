import pandas as pd

NUMERIC_INPUTS = [
    "pottery_load_kg", "target_temperature_c", "initial_moisture_pct",
    "planned_duration_hr", "insulation_rating", "kiln_volume_m3",
    "ambient_temperature_c"
]

def validate_input(data: dict) -> None:
    missing = [c for c in NUMERIC_INPUTS if c not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    for key in NUMERIC_INPUTS:
        value = data[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{key} must be numeric")
    if data["pottery_load_kg"] <= 0 or data["planned_duration_hr"] <= 0 or data["kiln_volume_m3"] <= 0:
        raise ValueError("load, duration and kiln volume must be greater than zero")
    if not 0 <= data["initial_moisture_pct"] <= 100:
        raise ValueError("initial_moisture_pct must be between 0 and 100")
    if not 0 < data["insulation_rating"] <= 1:
        raise ValueError("insulation_rating must be in (0, 1]")
    if not 100 <= data["target_temperature_c"] <= 1500:
        raise ValueError("target_temperature_c must be between 100 and 1500")

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["thermal_load_index"] = (
        out["pottery_load_kg"] * out["target_temperature_c"] *
        out["planned_duration_hr"] / (out["kiln_volume_m3"] * 1000)
    )
    out["moisture_removal_load"] = (
        out["pottery_load_kg"] * out["initial_moisture_pct"] / 100
    )
    return out
