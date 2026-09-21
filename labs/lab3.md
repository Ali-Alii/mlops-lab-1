# Lab 3: MLflow Registry, FastAPI, and Docker

## Completed result

- Registered model: `food11`
- Registered version: `1`
- Production alias: `champion`
- Best run: `powerful-bee-281`
- Docker image: `alialiii/food11-api`
- API endpoints: `GET /`, `GET /health`, and `POST /predict`

## Q1. Registered model versus logged model

MLflow assigned the model **version 1**. A logged model belongs to one training
run and stores that run's model artifact. A registered model gives selected
logged models a stable name, independent version numbers, aliases, tags, and a
promotion history suitable for deployment.

## Q2. Aliases and versions

Current MLflow uses user-defined aliases such as `champion` and `challenger`
instead of the deprecated built-in Staging/Production stages. Model versions
preserve immutable deployment candidates even when they came from different
runs. An alias is a mutable pointer: promoting a new version only requires
moving `champion`; application configuration does not need to change.

## Q3. Why use an MLflow model URI?

`models:/food11@champion` resolves the approved model through the registry and
loads its MLflow flavor, dependencies, metadata, and prediction interface. A
raw `.pth` file has none of that registry context and requires custom model
construction code. To serve a newer version, promote it by moving the
`champion` alias. For this self-contained Docker deployment, the resolved
champion is exported at build time and loaded from `/app/model`.

## Q4. Docker cache ordering

Dependency files are copied and installed before application source so Docker
can reuse the expensive dependency layer. Editing only `serve.py` invalidates
the small source-copy layer and later layers, while the virtual environment
remains cached.

## Q5. Image size and layers

Docker Desktop reports approximately **2.63 GB unpacked** for the completed
CPU-only image (the compressed image data is much smaller). Its largest content
is the Python environmentâ€”MLflow, CPU PyTorch, torchvision, and their
dependenciesâ€”followed by the roughly 45 MB model. A naive single-stage build
would additionally retain `uv`, download caches, and build-only files. `docker
history food11-api:latest` shows the virtual-environment copy as the largest
application layer. Multi-stage construction keeps builder contents out of the
runtime image.

## Q6. Why `.dockerignore` matters

Without it, Docker must send the Git history, virtual environment, MLflow
artifacts, DVC data, and caches as build context. That makes builds slower,
larger, less cache-friendly, and risks leaking local files. Copying the local
`.venv` can also break the Linux image because it was created for Windows.

## Q7. Container networking

Inside a container, `127.0.0.1` means the container itself, not the Windows
host. On Docker Desktop, `host.docker.internal` resolves to the host-side
address. This image does not require MLflow at runtime because the champion
model is bundled, but the development service can still use an overridden
`MLFLOW_TRACKING_URI` when loading directly from the registry.

## Q8. Restarting from the same image

Yes. A newly created container loads the same model without rebuilding because
the exported champion artifact and the Python environment are baked into the
image. Only uploaded request data is supplied at runtime.

## Q9. Making the image available elsewhere

The missing step is publishing the image to a container registry with a stable,
immutable version tag or digest. This lab publishes both `latest` and `1.0.2`
to Docker Hub under `alialiii/food11-api`. Deployments should prefer the
version tag or image digest rather than relying only on mutable `latest`.

## Useful commands

```powershell
# Select the best run and move the champion alias
uv run python src/food11/promote.py

# Export the current champion into the Docker build context
uv run python src/food11/export_champion.py

# Build and run
docker build -t food11-api:latest .
docker run --name food11-api -p 8000:8000 food11-api:latest

# Test
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:8000/predict
```
