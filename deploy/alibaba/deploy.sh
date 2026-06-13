#!/usr/bin/env bash
# Kioku v1 — deploy to Alibaba Cloud ECS with Docker.
#
# Two modes:
#   ./deploy.sh local                 build + run on this machine (docker compose)
#   ./deploy.sh ecs <user@host>       copy the repo to an ECS instance and run it
#
# Prereqs: Docker + docker compose; a .env at the repo root with a real
# QWEN_API_KEY (Alibaba Cloud Model Studio). The image bundles the Rust kiokud
# daemon, the FastAPI engine, and the web arena on port 8000.
#
# Provision the ECS instance first (Alibaba Cloud console or aliyun CLI):
#   - Ubuntu 22.04, an instance with >=2 GiB RAM (e.g. ecs.t6-c1m2.large)
#   - Security group: allow inbound TCP 8000 (and 22 for SSH)
#   - Install Docker:  curl -fsSL https://get.docker.com | sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-local}"

require_env() {
  if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "ERROR: $REPO_ROOT/.env not found. cp .env.example .env and set QWEN_API_KEY." >&2
    exit 1
  fi
  if grep -q "your-qwen-key" "$REPO_ROOT/.env"; then
    echo "WARNING: .env still has the placeholder QWEN_API_KEY — the demo will 502 until you set a real key." >&2
  fi
}

case "$MODE" in
  local)
    require_env
    echo ">> Building and starting Kioku locally (docker compose)…"
    cd "$REPO_ROOT"
    docker compose up --build -d
    echo ">> Up. Arena + API: http://localhost:8000   (health: /api/health)"
    docker compose ps
    ;;

  ecs)
    HOST="${2:-}"
    if [ -z "$HOST" ]; then echo "usage: ./deploy.sh ecs <user@host>" >&2; exit 1; fi
    require_env
    REMOTE_DIR="/opt/kioku"
    echo ">> Shipping Kioku to $HOST:$REMOTE_DIR …"
    ssh "$HOST" "sudo mkdir -p $REMOTE_DIR && sudo chown \$(whoami) $REMOTE_DIR"
    # Sync the source (excluding local artifacts); .env carries the key.
    rsync -az --delete \
      --exclude '.git' --exclude '.venv' --exclude 'kioku_data' \
      --exclude '__pycache__' --exclude '*.disk' \
      "$REPO_ROOT/" "$HOST:$REMOTE_DIR/"
    echo ">> Building and starting on the ECS instance…"
    ssh "$HOST" "cd $REMOTE_DIR && docker compose up --build -d && docker compose ps"
    echo ">> Live on the instance's public IP, port 8000. Open the security group if needed."
    ;;

  *)
    echo "usage: ./deploy.sh [local|ecs <user@host>]" >&2
    exit 1
    ;;
esac
