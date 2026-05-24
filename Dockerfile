FROM python:3.10-slim

WORKDIR /app

# 1. Cài đặt hệ thống (Layer này hiếm khi thay đổi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential supervisor && \
    rm -rf /var/lib/apt/lists/*

# 2. Cài đặt Python dependencies (Layer này chỉ thay đổi khi bạn sửa requirements.txt)
# Copy trước để tận dụng cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Copy source code (Layer này thường xuyên thay đổi)
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 9000 5000

COPY supervisord.conf /etc/supervisord.conf
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]