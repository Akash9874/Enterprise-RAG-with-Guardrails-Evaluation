# One CPU-only image serves both the API and the Streamlit UI (ADR-028).
FROM python:3.12-slim

# Pinned to the uv version this lockfile was produced with.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models/huggingface \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so a source edit does not reinstall torch. Linux torch resolves from
# the PyTorch CPU index (pyproject [tool.uv.sources]): no CUDA packages in the image.
COPY pyproject.toml uv.lock ./
# The uv download cache lives in a BuildKit cache mount, never in a layer. Measured without
# it: 1.75 GB of cached wheels shipped inside the image, beside the 2.1 GB venv (ADR-028).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group ui --no-install-project

COPY src ./src
COPY config ./config
COPY ui ./ui
COPY scripts ./scripts
COPY eval/adversarial ./eval/adversarial
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group ui

EXPOSE 8000 8501
CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
