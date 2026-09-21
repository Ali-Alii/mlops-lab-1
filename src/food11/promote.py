"""Promote the best Food-11 run to the MLflow Model Registry."""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="food11")
    parser.add_argument("--metric", default="val_accuracy")
    parser.add_argument("--model-name", default="food11")
    parser.add_argument("--alias", default="champion")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    return parser.parse_args()


def promote_best_model(args: argparse.Namespace) -> tuple[str, str, str, float]:
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {args.experiment}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.status = '{RunStatus.to_string(RunStatus.FINISHED)}'",
        order_by=[f"metrics.{args.metric} DESC"],
        max_results=1,
    )
    if not runs or args.metric not in runs[0].data.metrics:
        raise RuntimeError(f"No completed run contains metric {args.metric!r}")

    best_run = runs[0]
    score = best_run.data.metrics[args.metric]
    logged_models = [
        model
        for model in client.search_logged_models([experiment.experiment_id])
        if model.source_run_id == best_run.info.run_id
    ]
    if not logged_models:
        raise RuntimeError(f"Best run {best_run.info.run_id} has no logged model")
    logged_model = max(logged_models, key=lambda model: model.creation_timestamp)

    try:
        client.get_registered_model(args.model_name)
    except MlflowException:
        client.create_registered_model(
            args.model_name,
            description="Champion Food-11 ResNet-18 image classifier",
        )

    versions = client.search_model_versions(f"name='{args.model_name}'")
    matching = [
        version
        for version in versions
        if version.tags.get("logged_model_id") == logged_model.model_id
    ]
    if matching:
        version = matching[0]
    else:
        version = client.create_model_version(
            name=args.model_name,
            source=f"models:/{logged_model.model_id}",
            run_id=best_run.info.run_id,
            tags={"logged_model_id": logged_model.model_id},
            description=f"Selected by highest {args.metric} ({score:.6f})",
            model_id=logged_model.model_id,
        )

    for candidate in client.search_model_versions(f"name='{args.model_name}'"):
        client.set_model_version_tag(
            args.model_name,
            candidate.version,
            "champion",
            str(candidate.version == version.version).lower(),
        )
    client.set_registered_model_alias(args.model_name, args.alias, version.version)
    client.set_registered_model_tag(args.model_name, "selection_metric", args.metric)
    client.set_registered_model_tag(args.model_name, "champion_version", version.version)
    client.set_logged_model_tags(
        logged_model.model_id,
        {
            "champion": "true",
            "selection_metric": args.metric,
            "selection_score": str(score),
        },
    )
    client.set_tag(best_run.info.run_id, "champion", "true")
    return version.version, logged_model.model_id, best_run.info.run_id, score


def main() -> None:
    args = parse_args()
    version, model_id, run_id, score = promote_best_model(args)
    print(f"registered_model={args.model_name}")
    print(f"version={version}")
    print(f"alias={args.alias}")
    print(f"logged_model_id={model_id}")
    print(f"source_run_id={run_id}")
    print(f"{args.metric}={score:.6f}")


if __name__ == "__main__":
    main()
