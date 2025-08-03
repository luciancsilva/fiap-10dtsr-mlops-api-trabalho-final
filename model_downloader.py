from mlflow.tracking import MlflowClient
import mlflow
import json
from datetime import datetime

print("Downloading the latest model version...")

mlflow.set_tracking_uri("https://dagshub.com/luciancsilva/fiap-10dtsr-mlops-trabalho-final.mlflow")

model_name = "credit_score_model"
artifact_relative_path = "model/model.pkl"

client = MlflowClient()

versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max(versions, key=lambda v: int(v.version))

download_path = client.download_artifacts(
    run_id=latest_version.run_id,
    path=artifact_relative_path,
    dst_path="."
)

print(f"Latest model version: {latest_version.version}")
print(f"Model run ID: {latest_version.run_id}")

print(f"Writing model metadata...")

model_metadata = {
    "model_name": model_name,
    "version": latest_version.version,
    "run_id": latest_version.run_id,
    "source": latest_version.source,
    "downloaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open("model/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

print(f"Latest model downloaded successfully in path {download_path}")