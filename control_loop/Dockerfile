FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Copy thư viện cần thiết (có thể tạo requirements.txt riêng cho control_loop)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    requests \
    numpy \
    prometheus_client \
    huggingface_hub \
    pandas \
    joblib \
    tensorboardX

# Copy thư mục control_loop
COPY ./control_loop /app/control_loop

# CỰC KỲ QUAN TRỌNG: Copy rl_engine vào container để import được
COPY ./rl_engine /app/rl_engine
COPY ./llm /app/llm

# Thiết lập PYTHONPATH để Python nhận diện được module ở thư mục gốc
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000

CMD ["python", "control_loop/main_loop.py"]