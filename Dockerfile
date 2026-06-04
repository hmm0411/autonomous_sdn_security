FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . /app

RUN mkdir -p /app/logs /app/models /app/results /app/mlruns

EXPOSE 8000 8001 8501 9002 9003 9100

CMD ["tail", "-f", "/dev/null"]