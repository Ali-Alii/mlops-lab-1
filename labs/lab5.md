# Lab 5: CI/CD workflow

```text
git push / pull request
    |
    +-- automatic: GitHub Actions test job
    |      uv sync --frozen -> Ruff -> unit/API/data-contract tests
    |
    +-- automatic after tests: smoke job
    |      Docker build -> container with SKIP_MODEL=1 -> /health
    |
    +-- automatic after smoke, main pushes only: push-image job
           GHCR image tag = full Git commit SHA (code version)

human: review/merge the pull request; deploy the chosen image SHA
human or a separately triggered training workflow: train and register model
quality gate: compare candidate test accuracy with champion + 2 standard errors
    pass -> MLflow champion alias (model version)
    fail -> MLflow challenger alias
running service: load models:/food11@champion from MLflow at startup
```

The CI runner has no DVC dataset or MLflow server. Its tests use a tiny Git
fixture and a fake model; it does not train or evaluate the real model. Run
`validate_data.py` after `dvc pull` on a machine with the data, before training.
The current mini dataset fails the cross-split duplicate check, so its test
accuracy should not be treated as a clean evaluation until the data is fixed
and the model is retrained.

The GHCR image built by GitHub contains code and dependencies but not the
Git-ignored bundled model export. For real serving, set `MLFLOW_TRACKING_URI`
and `MLFLOW_MODEL_URI=models:/food11@champion` to a reachable registry. The
`SKIP_MODEL=1` smoke mode only proves that the API container starts and answers
`/health`; it is not a prediction test.
