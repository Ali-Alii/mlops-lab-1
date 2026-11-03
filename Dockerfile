FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_MODEL_URI="/app/deployment/champion-model"

COPY --from=builder /app/.venv /app/.venv
COPY src /app/src
COPY deployment /app/deployment

EXPOSE 8000
CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]

