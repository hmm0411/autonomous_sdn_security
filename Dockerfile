# 1. Base image siêu nhẹ
FROM python:3.10-slim

# 2. Thư mục làm việc và User non-root
WORKDIR /app
RUN useradd -m -r appuser && chown appuser /app

# 3. Cài đặt hệ thống (Xóa sạch supervisor, git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. Tạo trước các thư mục cần thiết và cấp quyền
RUN mkdir -p /app/logs /app/models && \
    chown -R appuser:appuser /app/logs /app/models

# 5. Cài đặt Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 6. Copy source code
COPY --chown=appuser:appuser . /app/

# 7. Biến môi trường
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8001 9002 9003

# 8. Chạy bằng User bảo mật
USER appuser

# Lệnh chờ mặc định (Dành cho các container Agent Standby)
# Còn container Serving thì file docker-compose.yml đã có entrypoint riêng rồi
CMD ["tail", "-f", "/dev/null"]