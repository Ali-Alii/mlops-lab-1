# Lab 5 Answers: CI/CD

## Q1. Why not load the real model in a unit test?

It would make the test depend on MLflow availability, credentials, network
access, model artifacts, and CPU/GPU resources. Any of those can fail even
when image preprocessing and API code are correct. The fake model keeps the
unit test fast and deterministic.

## Q2. What does the fake-model API test prove?

It proves the API accepts an image upload, decodes and preprocesses it into
the expected `(1, 3, 128, 128)` tensor, calls the model, and returns a valid
prediction JSON with `predicted_category`, confidence, and top three classes.
It does **not** prove the real model is accurate, that MLflow is available, or
that the model artifact loads correctly.

## Q3. Which check catches a train/test duplicate?

The cross-split content-hash check catches it even when the filenames differ.
A duplicate leaks evaluation examples into training, making accuracy look
better than performance on unseen data. A corrupted image usually fails
loudly; leakage can silently invalidate the evaluation and promotion gate.
The validator's synthetic test deliberately copies a training image into the
evaluation split and confirms failure.

## Q4. Where can data validation run if GitHub CI has no DVC data?

Run it after `dvc pull` on a data-accessible training machine, before any
training, and again in a deployment/pre-release job that can access the DVC
remote. This project also calls it at the start of `train.py`. The cloud CI
tests the validator itself with a small synthetic dataset rather than pulling
the private/full dataset.

## Q5. What does `needs: test` do?

The Docker smoke job starts only after `test` succeeds. If lint or a unit
test fails, smoke is skipped. Without `needs`, both jobs could run in parallel,
wasting build time and potentially displaying a green smoke check beside a
red test check.

## Q6. Run duration and cache comparison

The first [GitHub Actions run](https://github.com/Ali-Alii/mlops-lab-1/actions/runs/41634047511)
passed in **5 minutes 30 seconds**; the [second run](https://github.com/Ali-Alii/mlops-lab-1/actions/runs/41634950611)
passed in **1 minute 56 seconds**. The `test` job changed only from 18 to 17
seconds. The main improvement was in image publishing: the second build could
reuse the GitHub Actions BuildKit cache from the first run. UV also caches
dependencies for `test`. The `smoke` job uses a plain `docker build` on a
fresh runner and does not share the publishing job's BuildKit cache.

## Q7. Why tag the image with the commit SHA?

The SHA identifies one immutable code revision, so the image can be traced to
the source that built it. `latest` is mutable: machine A may pull before a new
push and machine B afterward, leaving both apparently on `latest` but running
different code.

## Q8. Why explicit `permissions`?

The test and smoke jobs get read-only repository access; only the publish job
gets `packages: write` so it can push to GHCR. A compromised step with broad
write access could alter code, releases, workflow files, or packages. Least
privilege limits that damage.

## Q9. What does a failing PR check change?

The PR shows a failed `test` check and the dependent smoke job is skipped.
If branch protection requires both checks, merging is blocked until the
failure is fixed. This makes the rule consistent for every team member,
instead of trusting each person to remember to run tests locally. Branch
protection and a deliberately failing PR still need to be configured and
demonstrated in GitHub; they are not proven by the local tests.

## Q10. What did the quality gate decide?

For a local demonstration, we registered two **existing** training runs as
candidate model versions 2 and 3. The existing champion is version 1 with
`test_accuracy=0.650547`. The mini evaluation split has `n=1096` images,
giving `2 * se = 0.028804` (2.88 percentage points) and a promotion
threshold of `0.679352`.

| Candidate | Test accuracy | Decision |
| --- | ---: | --- |
| Version 2 | 0.633212 | `challenger` |
| Version 3 | 0.291971 | `challenger` |

The champion remains version 1. The gate wrote its decision and scores as
model-version tags. These are **not** new Lab 5 training runs. More
importantly, validation found 43 cross-split duplicate images in the current
mini dataset, so these historical accuracy numbers should not be used as a
trusted production promotion decision. Clean the dataset, retrain, and rerun
the gate before relying on it.

## Q11. How could promotion run automatically after training?

Run training and `promote.py` sequentially on a self-hosted runner that can
reach the local MLflow server, or move MLflow to an accessible hosted service
such as DagsHub and invoke promotion from a post-training CI job with scoped
credentials. Neither should promote while the data contract fails.

## Q12. What is the pipeline?

See [the pipeline diagram](../labs/lab5.md). Push/PR automatically runs UV
sync, Ruff, unit tests, Docker build, and a container health smoke test. A
successful `main` push automatically publishes a GHCR image tagged with the
Git commit SHA (the code version). A human still reviews/merges, chooses when
to deploy, and triggers clean-data training. The quality gate chooses the
MLflow `champion` alias (the model version); the running service loads that
version when it starts or restarts.

## Reproduce the local checks

```powershell
uv run ruff check src tests
uv run pytest -q
uv run python src/food11/validate_data.py data/food11_processed_mini
```

The last command currently exits with status 1 because the duplicate check
correctly fails. Do not weaken that check to make the command green.
