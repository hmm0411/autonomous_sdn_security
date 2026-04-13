FROM python:3.10-slim
WORKDIR /app
COPY . /app/rl_engine/
RUN pip install --upgrade pip
RUN pip install -r /app/rl_engine/requirements.txt
ENV PYTHONPATH="/app"
