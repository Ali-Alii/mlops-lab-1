FROM python:3.13-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /bin/uv
ENV UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=10
RUN --mount=type=cache,id=food11-uv-py313,target=/root/.cache/uv,sharing=locked \
    uv pip install --system streamlit requests

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
