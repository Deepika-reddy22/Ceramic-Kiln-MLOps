import os, json, joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pipeline import engineer_features

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
FEATURES = ["pottery_load_kg","target_temperature_c","initial_moisture_pct",
"planned_duration_hr","insulation_rating","kiln_volume_m3","ambient_temperature_c",
"thermal_load_index","moisture_removal_load"]

def main():
    frames = [pd.read_csv(os.path.join(DATA, f"kiln_batches_v{i}.csv")) for i in (1,2)]
    df = engineer_features(pd.concat(frames, ignore_index=True))
    df[["batch_id"] + FEATURES + ["energy_consumption_kwh"]].to_csv(
        os.path.join(DATA, "feature_store_materialized.csv"), index=False)
    X, y = df[FEATURES], df["energy_consumption_kwh"]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42)
    models = {"LinearRegression": LinearRegression(),
              "RandomForestRegressor": RandomForestRegressor(
                  n_estimators=300,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1)}
    results = {}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ceramic-kiln-energy")
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr,ytr)
            pred = model.predict(Xte)
            scores = {"mae":mean_absolute_error(yte,pred),
                      "rmse":mean_squared_error(yte,pred)**0.5,
                      "r2":r2_score(yte,pred)}
            results[name] = scores
            mlflow.log_params({"model":name,"dataset_versions":"v1+v2","test_size":0.2})
            mlflow.log_metrics(scores)
            mlflow.sklearn.log_model(model, artifact_path="model")
    winner = min(results, key=lambda name: results[name]["mae"])
    model = models[winner]
    model.fit(X,y)
    os.makedirs(os.path.join(ROOT,"artifacts"),exist_ok=True)
    joblib.dump({"model":model,"features":FEATURES,"model_name":winner},
                os.path.join(ROOT,"artifacts","kiln_model.joblib"))
    with open(os.path.join(ROOT,"artifacts","metrics.json"),"w") as f:
        json.dump({"winner":winner,"results":results},f,indent=2)
    # Register the winning model from the final training run in the MLflow Model Registry.
    with mlflow.start_run(run_name="register-"+winner):
        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        try:
            mlflow.register_model(model_uri, "CeramicKilnEnergyModel")
        except Exception as exc:
            print("Registry registration requires a registry-capable MLflow backend:", exc)
    print(json.dumps({"winner":winner,"results":results},indent=2))

if __name__ == "__main__":
    main()
