#!/usr/bin/env sh
# Start the Cadran daemon, then the engine. The engine connects to the running
# kiokud socket (or falls back to PyStore if the daemon is unavailable).
set -e

echo "kioku: starting kiokud daemon"
KIOKUD_SOCKET="${KIOKU_SOCKET:-/tmp/kiokud.sock}" \
KIOKUD_DISK="${KIOKU_DATA_DIR:-/data}/kioku_box.disk" \
  /app/substrate/kiokud &

# Give the daemon a moment to bind its socket.
i=0
while [ ! -S "${KIOKU_SOCKET:-/tmp/kiokud.sock}" ] && [ "$i" -lt 50 ]; do
  i=$((i + 1)); sleep 0.1
done

echo "kioku: starting FastAPI engine on :8000"
exec python -m uvicorn engine.main:get_app --factory --host 0.0.0.0 --port 8000
