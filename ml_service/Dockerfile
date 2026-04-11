FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/cache/hf \
    HF_HOME=/app/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl wget git \
    chromium-driver chromium \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
