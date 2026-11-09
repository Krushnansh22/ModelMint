# ModelMint backend.
#
# Build targets:
#   slim  (default) -- sklearn + ONNX only
#   full            -- adds xgboost/lightgbm/catboost/torch
#   dl              -- full + TensorFlow/Keras/TFLite/Hugging Face
#   gguf            -- full + llama-cpp-python
#
#   docker build --target full -t modelmint-backend:full backend/
#   docker build --target dl   -t modelmint-backend:dl   backend/
#   docker build --target gguf -t modelmint-backend:gguf backend/
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

RUN useradd --create-home --uid 10001 modelmint \
    && mkdir -p /app/storage \
    && chown -R modelmint:modelmint /app

USER modelmint
EXPOSE 8000

# One worker: the model cache and in-memory rate limiter are per-process.
# Use Redis for shared limits and scale out with a shared serving tier.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


FROM base AS slim


FROM base AS full
USER root
COPY requirements-formats.txt ./
RUN pip install -r requirements-formats.txt
USER modelmint


FROM full AS dl
USER root
COPY requirements-formats.txt requirements-dl.txt ./
RUN pip install -r requirements-dl.txt
USER modelmint


FROM full AS gguf
USER root
COPY requirements-gguf.txt ./
RUN pip install -r requirements-gguf.txt
USER modelmint
