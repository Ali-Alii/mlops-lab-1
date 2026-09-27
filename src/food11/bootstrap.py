"""Seed an empty Compose MLflow registry from the model bundled in Lab 3."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "food11"
ALIAS = "champion"
MODEL_PATH = Path("/app/model")


def wait_for_tracking_server(client: MlflowClient) -> None:
    for attempt in range(30):
        try:
            client.search_registered_models(max_results=1)
            return
        except Exception:
            if attempt == 29:
                raise
            time.sleep(2)


def register_bundled_model(client: MlflowClient) -> str:
    if not (MODEL_PATH / "MLmodel").is_file():
        raise FileNotFoundError(f"Bundled MLflow model missing: {MODEL_PATH}")

    mlflow.set_experiment("food11-compose")
    with mlflow.start_run(run_name="lab4-bundled-champion") as run:
        mlflow.log_artifacts(str(MODEL_PATH), artifact_path="model")
        source = mlflow.get_artifact_uri("model")

    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(MODEL_NAME)

    version = client.create_model_version(
        name=MODEL_NAME,
        source=source,
        run_id=run.info.run_id,
        description="Imported from the Lab 3 champion image",
    )
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    return version.version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-version",
        action="store_true",
        help="Register the bundled model again and move champion to its new version.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()
    wait_for_tracking_server(client)

    if not args.new_version:
        try:
            current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            print(f"Using existing {MODEL_NAME}@{ALIAS} version {current.version}", flush=True)
            return
        except MlflowException:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if versions:
                latest = max(versions, key=lambda item: int(item.version))
                client.set_registered_model_alias(MODEL_NAME, ALIAS, latest.version)
                print(f"Restored {ALIAS} alias to version {latest.version}", flush=True)
                return

    version = register_bundled_model(client)
    print(f"Registered {MODEL_NAME} version {version} as {ALIAS}", flush=True)


if __name__ == "__main__":
    main()
