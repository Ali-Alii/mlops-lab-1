"""Export the registry champion into the Docker build context."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default="models:/food11@champion")
    parser.add_argument("--output", type=Path, default=Path("deployment/champion-model"))
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    downloaded = Path(mlflow.artifacts.download_artifacts(args.model_uri))
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded, args.output)
    print(f"Exported {args.model_uri} to {args.output}")


if __name__ == "__main__":
    main()

