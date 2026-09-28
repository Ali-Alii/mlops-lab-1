# Lab 4 Answers: Docker Compose

This project runs three services: MLflow (tracking and model registry), FastAPI
inference, and a Streamlit upload page. They communicate on the private Docker
Compose network. The deployment uses the MLflow alias `champion` instead of the
lab's older `Staging` stage.

## Q1. What happens without a volume at `/mlflow-data`?

The MLflow database and artifacts are written into that container's writable
layer. Stopping and restarting **the same container** keeps them, but removing
it and creating a new container from the image loses them. The new MLflow UI
has an empty registry because images do not include changes made at runtime.

## Q2. Why a named volume? Would a bind mount work?

The named `mlflow-data` volume outlives the MLflow container and is managed by
Docker. It avoids a machine-specific path inside the repository and host file
permission issues. A bind mount would also persist the database and artifacts;
it is useful when you want to inspect the files directly on the host, but its
path and permissions must be managed yourself.

## Q3. Why does `http://mlflow:5000` resolve now?

Compose creates a private network and provides DNS for service names on it.
From the inference container, `mlflow` resolves to the MLflow container. In
Lab 3, the separate container was not on such a Compose network, so the
service name did not exist there and we used `host.docker.internal` to reach
MLflow running on the host.

## Q4. Why configure `INFERENCE_URL` with an environment variable?

The frontend image can run in different environments without changing its
code. Compose sets `INFERENCE_URL=http://inference:8000`; outside Compose, that
hostname may not exist, so we can supply another reachable URL when starting
the frontend container.

## Q5. Why is inference not published to the host?

Only a browser needs host access to the MLflow UI (port 5000) and Streamlit
page (port 8501). The frontend reaches FastAPI at
`http://inference:8000/predict` over the private Compose network, so no host
port mapping is needed for inference. This also avoids exposing the API
unnecessarily.

## Q6. What if MLflow is not ready when inference starts?

With start-order-only `depends_on`, the inference process could fail while
loading the model and the container could exit or restart. Check the cause
with `docker compose logs inference`. Our Compose file adds an MLflow health
check and waits for `service_healthy` before starting inference. The bootstrap
code also retries the tracking-server connection. These measures handle the
startup race, though a later outage can still affect the service.

## Q7. What does `docker compose ps` show for ports?

MLflow shows `127.0.0.1:5000->5000/tcp`; frontend shows
`127.0.0.1:8501->8501/tcp`; inference shows only its internal `8000/tcp`.
That matches `docker-compose.yml`: only the two browser-facing services have
`ports` entries. The `127.0.0.1` binding also limits access to this computer.

## Q8. Does promoting a new model change a running prediction immediately?

No. FastAPI loads the model once when its process starts, so merely moving
the `champion` alias does not replace the model already in memory. Run
`docker compose restart inference`, then upload the image again. In our test,
the response changed to `model_version: "2"` after the restart. We registered
the same model weights as version 2 for this test, so the prediction values
were unchanged even though the served registry version changed.

## Q9. Why does restart work without rebuilding the image?

The inference image already contains the application, Python dependencies,
and a bundled model for bootstrapping an empty registry. At startup the app
resolves `models:/food11@champion` through MLflow and loads the version that
alias currently points to. Restarting repeats that lookup; rebuilding code or
the Docker image is unnecessary just to switch the registry alias.

## Q10. What survives `down`/`up`, and what changes with `down -v`?

`docker compose down` removes containers and the Compose network, but keeps
the named volume. In our real project, `champion` still pointed to version 2
after `down` followed by `up`. `docker compose down -v` also deletes the named
volume, including MLflow's SQLite database, registered model versions, and
artifacts. A separate disposable test project returned to version 1 after
that command because bootstrap created a fresh registry. We did **not**
delete the real project's MLflow volume.

## Q11. What is needed beyond single-machine Compose?

Running three reliable inference replicas needs an orchestrator that schedules
replicas, replaces failed instances, and routes requests through a load
balancer. MLflow surviving a machine failure needs its SQLite database and
local named volume replaced by durable shared services, such as a remote
database and object storage, with backups and suitable access controls.
Compose alone does not provide multi-host scheduling or high availability.

## How to inspect the result

```powershell
docker compose up -d --build
docker compose ps
docker compose logs inference
```

Open the upload page at <http://127.0.0.1:8501> and MLflow at
<http://127.0.0.1:5000>. In MLflow, open **Models**, choose **food11**, and
inspect versions and the `champion` alias. Upload a food image in the
frontend to see its predicted class, confidence, and model version.

Do not run `docker compose down -v` against the real project unless you intend
to delete its MLflow registry and artifacts.
