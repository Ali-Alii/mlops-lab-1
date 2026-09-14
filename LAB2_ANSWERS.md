# Lab 2 Answers: Training and MLflow

## Q1. What changed in `pyproject.toml` and `uv.lock`?

`pyproject.toml` now declares MLflow, PyTorch, torchvision, and scikit-learn as
direct project dependencies. `uv.lock` records exact resolved versions and all
transitive dependencies so every machine can reproduce the same environment.

## Q2. Backend store, artifact root, metadata, and artifacts

`--backend-store-uri sqlite:///mlflow.db` tells MLflow to store structured run
metadata in a local SQLite database. This includes experiments, run IDs,
parameters, metric values, timestamps, tags, and artifact locations.

`--default-artifact-root ./mlruns` tells MLflow where to store run artifacts.
Artifacts are files such as serialized models, environment specifications, and
other uploaded outputs. Metadata is searchable structured information; artifacts
are the larger files produced by a run.

## Q3. Why ignore `mlflow.db` and `mlruns/`?

They are generated local experiment outputs that change every time training is
run. Tracking them in Git would produce large, noisy commits and SQLite merge
conflicts. They should not be tracked by DVC either because MLflow already owns
their lifecycle and records the relationship between each run and its artifacts.

## Q4. What happens when a new experiment name is used?

`mlflow.set_experiment("food11")` creates the experiment automatically if it does
not exist, then makes it the active experiment. It appears in the MLflow UI as
soon as it is created.

## Q5. Parameters versus metrics

A parameter is fixed configuration recorded once for a run, such as learning
rate or batch size. A metric is a measured result, such as loss or accuracy,
that may change during training. Metrics accept a `step` so MLflow can store and
plot their history across epochs; parameters do not have a history.

## Q6. Where does the model artifact live?

The run page shows parameters, metric charts, and the logged `model`. With
MLflow 3.16, `mlflow.pytorch.log_model` creates a Logged Model linked to the run,
so it is visible under the run's **Models** output and the top-level **Models**
page rather than as a conventional run artifact. Its files live under the local
`mlruns/` artifact root. The SQLite database stores the model metadata and
location rather than the model bytes.

## Q7. Which learning rate performed best?

Among the tested runs, learning rate `0.001` with batch size 32 performed best,
with final validation accuracy `0.605839`. Higher learning rates are not always
better: `0.01` learned quickly but finished slightly lower, while `0.0001`
learned too slowly within the five-epoch budget.

## Q8. Parallel-coordinate pattern

The parallel-coordinates plot shows that `lr=0.001`, `batch_size=32` leads to the
best final validation accuracy. Keeping `lr=0.001` and increasing the batch size
to 64 reduced final accuracy from about `0.606` to `0.585`. The very low
`lr=0.0001` performed substantially worse in this short training budget.

## Q9. Best run ID

The best run is `powerful-bee-281`, with run ID
`f7b60acfcdab47f8916a808004b6c607` and final validation accuracy `0.605839`.

## Experiment results

| Run | Learning rate | Batch size | Final validation accuracy | Test accuracy | Run ID |
|---|---:|---:|---:|---:|---|
| powerful-bee-281 | 0.001 | 32 | 0.605839 | 0.650547 | `f7b60acfcdab47f8916a808004b6c607` |
| glamorous-sheep-248 | 0.01 | 32 | 0.601277 | 0.632299 | `68fe85488d8043ad9f47beb9eb75438e` |
| gaudy-pig-348 | 0.001 | 64 | 0.584854 | 0.633212 | `ebff4ce3eac94fc4b68066c236bd0a91` |
| mysterious-colt-920 | 0.0001 | 32 | 0.292883 | 0.291971 | `a16090d0eb26426fb3d523ff546078ee` |
