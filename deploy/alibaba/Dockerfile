# Kioku v1 — single image: builds the Rust kiokud daemon, serves the FastAPI
# engine + the web arena (same origin) on :8000. Designed for Alibaba Cloud ECS.
FROM python:3.11-slim AS base

# Rust toolchain (Debian's rustc supports --edition=2021) to build the substrate
# daemon, plus curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        rustc curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY engine/requirements.txt engine/requirements.txt
COPY engine/requirements-postgres.txt engine/requirements-postgres.txt
RUN pip install --no-cache-dir -r engine/requirements.txt \
                               -r engine/requirements-postgres.txt

# Source
COPY substrate/ substrate/
COPY engine/ engine/
COPY web/ web/
COPY pyproject.toml ./

# Build the headline path at image time so startup is instant.
RUN cd substrate && rustc --edition=2021 -C opt-level=3 kiokud.rs -o kiokud

ENV KIOKU_DATA_DIR=/data \
    KIOKU_SOCKET=/tmp/kiokud.sock \
    KIOKU_STORE=rust \
    KIOKU_WEB_ORIGIN=* \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]
EXPOSE 8000

COPY deploy/alibaba/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
