FROM python:3.12-slim AS wheels

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# llama-cpp-python may compile from source for a new Python platform. Build the
# wheel in a throwaway stage so compilers are not present in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOCAL_MODEL_PATH=/app/models/model.gguf \
    LOCAL_MODEL_THREADS=2

WORKDIR /app

COPY --from=wheels /wheels /wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && pip install --no-cache-dir /wheels/* \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /wheels

COPY . .

# A model is optional: place it at models/model.gguf before building to enable
# self-contained local inference.  Without it, use the harness Fireworks env.
RUN mkdir -p /input /output

CMD ["python", "main.py", "--batch"]
