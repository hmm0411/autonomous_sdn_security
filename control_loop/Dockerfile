FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    requests \
    numpy \
    pandas \
    prometheus_client \
    mlflow \
    joblib \
    scikit-learn \
    matplotlib \
    huggingface_hub \
    transformers

COPY ./control_loop /app/control_loop
COPY ./rl_engine /app/rl_engine
COPY ./digital_twin /app/digital_twin
COPY ./llm /app/llm

RUN mkdir -p /app/logs /app/models /app/results

EXPOSE 9100

CMD ["python", "control_loop/main_loop.py"]