# Deployment model

Run the following while the local MLflow server is available to export the
current registry champion into the Docker build context:

```powershell
uv run python ./src/food11/export_champion.py
```

The generated `deployment/champion-model/` directory is intentionally ignored
by Git. The Docker image packages this exact exported champion so inference is
self-contained and reproducible.

