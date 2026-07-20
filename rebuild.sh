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

# All topologies share the same host ports, so a plain `docker compose down`
# (which only tears down .env's currently active profiles) can leave a
# previous topology's containers running and collide on those ports if .env
# was reconfigured (e.g. re-running setup.sh to switch app -> mono) since the
# last rebuild. Tear down every known profile up front regardless of .env.
KNOWN_PROFILES=(app training mono dev dev-mono)
down_profile_args=()
for p in "${KNOWN_PROFILES[@]}"; do down_profile_args+=(--profile "$p"); done

echo "Stopping any running stack (all known profiles, to clear host ports)..."
docker compose "${down_profile_args[@]}" down --remove-orphans

echo "Rebuilding and starting..."
docker compose up -d --build

echo
docker compose ps
