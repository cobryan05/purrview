#!/usr/bin/env bash
#
# Rebuild and (re)start the stack described by .env.
#
# Topology, image base, and GPU wiring all live in .env (created by ./setup.sh).
# This just tears the current stack down and brings it back up rebuilt.
#
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "No .env found. Run ./setup.sh first (or copy .env.example to .env)." >&2
  exit 1
fi

# Keep bind-mount ownership aligned with the host user for this run.
export APPUSER_UID="$(id -u)"
export APPUSER_GID="$(id -g)"

echo "Stopping current stack (per .env profiles)..."
docker compose down --remove-orphans

echo "Rebuilding and starting..."
docker compose up -d --build

echo
docker compose ps
