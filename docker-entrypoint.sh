#!/bin/sh
# Mounted volumes/disks (Render, docker-compose) are owned by root, so fix
# ownership at start-up, then drop privileges to the unprivileged "app" user.
set -e
DATA_DIR="${DATA_DIR:-/data}"
mkdir -p "$DATA_DIR"
chown -R app:app "$DATA_DIR" 2>/dev/null || true
exec runuser -u app -- streamlit run app.py \
  --server.port="${PORT:-8501}" --server.address=0.0.0.0
