#!/usr/bin/env bash
# Smoke-test the prediction API. Tries host :5002 first (same as guide.md); if that fails
# (e.g. shell is a dev container), falls back to docker compose exec against in-container :5000.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT/docker/compose.yaml"
cd "$ROOT"

try_host() {
  local base="$1"
  curl -fsS --connect-timeout 2 "${base}/api/health" >/dev/null 2>&1
}

GW=""
if command -v ip >/dev/null 2>&1; then
  GW="$(ip route 2>/dev/null | awk '/default/ {print $3}' | head -1 || true)"
fi

WORKING_BASE=""
for base in "http://127.0.0.1:5002" "http://localhost:5002"; do
  if try_host "$base"; then
    WORKING_BASE="$base"
    break
  fi
done
if [ -z "$WORKING_BASE" ] && [ -n "$GW" ]; then
  if try_host "http://${GW}:5002"; then
    WORKING_BASE="http://${GW}:5002"
  fi
fi

json_tool() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -m json.tool
  else
    python -m json.tool
  fi
}

if [ -n "$WORKING_BASE" ]; then
  echo "# OK: host can reach Docker-published port (same as guide.md curl examples)"
  echo "#    ${WORKING_BASE}/api/health"
  curl -sS "${WORKING_BASE}/api/health"
  echo
  # Valid catalog pair from data/train_events.csv (CustomerID=11000, ProductID=771)
  curl -sS "${WORKING_BASE}/api/purchase?customer_id=11000&product_id=771" | json_tool
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not in PATH; cannot fall back to compose exec." >&2
  exit 1
fi

if ! docker compose -f "$COMPOSE_FILE" ps -q api 2>/dev/null | grep -q .; then
  echo "API container is not running. From assignment2 root run:" >&2
  echo "  docker compose -f docker/compose.yaml up --build" >&2
  exit 1
fi

echo "# Host :5002 not reachable from this shell (common in dev containers)."
echo "# Using: docker compose exec api curl http://127.0.0.1:5000/... (in-container port)"
docker compose -f "$COMPOSE_FILE" exec -T api curl -sS "http://127.0.0.1:5000/api/health"
echo
docker compose -f "$COMPOSE_FILE" exec -T api curl -sS "http://127.0.0.1:5000/api/purchase?customer_id=11000&product_id=771" | json_tool
