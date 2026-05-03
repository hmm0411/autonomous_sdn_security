#!/bin/bash
set -e

mkdir -p /app/logs
mkdir -p /app/models

echo "Starting RL inference API..."
exec /usr/bin/supervisord -c /etc/supervisord.conf