FROM python:3.10-slim
WORKDIR /app
COPY . /app/rl_engine/
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy pandas scikit-learn gymnasium tensorboard matplotlib seaborn requests mlflow prometheus-client
ENV PYTHONPATH="/app"
