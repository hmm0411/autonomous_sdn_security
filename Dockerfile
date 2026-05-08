FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/rl_engine/requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy pandas scikit-learn gymnasium tensorboard matplotlib seaborn requests mlflow prometheus-client

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 9000 5000

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisord.conf

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Run entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]