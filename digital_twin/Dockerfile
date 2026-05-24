FROM python:3.10-slim

WORKDIR /app

# Cài đặt dependency hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
# Việc copy và pip install trước giúp Docker cache layer này lại
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Port của ứng dụng (nếu bạn chạy Flask/Metrics trên cổng 8000, 9002, 9003 thì cần EXPOSE đủ)
EXPOSE 8000 8001 9002 9003

# Entrypoint tùy theo nhu cầu
CMD ["python", "main.py"]