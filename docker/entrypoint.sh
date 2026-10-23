#!/bin/sh
# The upstream Hermes image starts a long-lived gateway from its own entrypoint.
# This harness runs the Hermes CLI as an isolated subprocess instead, so bypass
# that supervisor and execute the harness API or worker directly.
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec /opt/harness-venv/bin/uvicorn clinical_coder.api:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-8080}"
