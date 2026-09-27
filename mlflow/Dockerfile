FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /bin/uv
ENV UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=10
RUN --mount=type=cache,id=food11-uv-py313,target=/root/.cache/uv,sharing=locked \
    uv pip install --system mlflow==3.16.0

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", "--workers", "1", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/artifacts", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]
