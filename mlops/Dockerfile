FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY mlops/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlops/ .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 5001

CMD ["python", "main.py"]