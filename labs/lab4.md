# Lab 4: Docker Compose

## What was built

`docker-compose.yml` starts three services on one private Compose network:

```text
Browser :8501 → Streamlit frontend → http://inference:8000/predict
                                      → MLflow model loaded from http://mlflow:5000
Browser :5000 → MLflow UI
```

The inference service loads `models:/food11@champion` at startup. If the
Compose registry is empty, `src/food11/bootstrap.py` uploads the Lab 3 model
bundled in the inference image, registers `food11` version 1, and sets the
`champion` alias. The MLflow SQLite database and artifacts live in the named
volume `food11-lab4_mlflow-data`.

## Answers

1. Without a volume, MLflow writes to the container's writable layer. Stopping
   and restarting that same container retains it, but removing the container
   loses the database and artifacts. A new container from the image starts
   empty.

2. A named volume keeps state independently of containers and avoids tying the
   Compose file to a host directory or host filesystem permissions. A bind
   mount would also persist data and is useful for local inspection, but its
   host path must exist and be portable across machines.

3. Compose creates a private network and DNS records for its service names.
   Therefore `mlflow` resolves from the inference container as
   `http://mlflow:5000`. Outside that network, the name does not resolve.

4. `INFERENCE_URL` lets the same frontend image point at
   `http://inference:8000` inside Compose or another URL when run on its own.

5. `mlflow` publishes host port 5000 for the MLflow UI and `frontend` publishes
   host port 8501 for the upload page. `inference` shows only `8000/tcp` in
   `docker compose ps`; the frontend calls it directly on the private network.

6. Start order alone does not ensure readiness. If inference tries to load the
   registry model before MLflow accepts connections, startup fails and the
   container may exit or restart. This Compose file adds MLflow and inference
   health checks plus `depends_on: condition: service_healthy`. The bootstrap
   script also retries the tracking-server connection.

7. Observed `docker compose ps`: MLflow had `127.0.0.1:5000->5000/tcp`,
   frontend had `127.0.0.1:8501->8501/tcp`, and inference had `8000/tcp` only.

8. The inference process keeps its model in memory. Moving `champion` to
   version 2 did not change the loaded model until `docker compose restart
   inference`. After restart, `/predict` returned `model_version: "2"`.
   The prediction values stayed the same because the test registered the same
   weights as a second version.

9. Restart works because the inference service resolves the registry alias and
   downloads its artifacts from MLflow at startup. The image already has the
   application and its Python libraries; changing an MLflow alias does not
   require an image rebuild.

10. On the real project, `docker compose down` followed by `docker compose up
    -d` retained `champion` version 2. A separate disposable project was
    promoted to version 2, then run through `down -v` and `up`; it started
    again at version 1. `-v` deletes the named MLflow volume, including its
    database, model versions, and artifacts. The disposable project and volume
    were removed after the test; the real project volume was preserved.

11. Compose operates on one Docker host. Three reliable inference replicas
    would need a load balancer, orchestration with replica scheduling and
    health-based replacement, and a shared model source. MLflow surviving a
    machine failure would need a remote database and durable shared artifact
    storage rather than this host's SQLite database and named volume.

## Run and inspect

```powershell
docker compose up -d --build
docker compose ps
docker compose logs -f inference
```

Open the [frontend](http://127.0.0.1:8501) and
[MLflow UI](http://127.0.0.1:5000). To demonstrate a version change with the
same bundled weights:

```powershell
docker compose exec inference python -m src.food11.bootstrap --new-version
docker compose restart inference
```

Use `docker compose down` to stop containers while retaining the MLflow data.
Avoid `docker compose down -v` on the real project unless you intend to erase
the registry and artifacts.
