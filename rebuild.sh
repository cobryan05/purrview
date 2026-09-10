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

# The Dockerfile uses BuildKit-only syntax (--mount=type=cache). Without the
# buildx plugin, `docker compose build` silently falls back to the legacy
# builder and fails deep into the build with an opaque "--mount option
# requires BuildKit" error. Catch it here instead, with the actual fix.
if ! docker buildx version >/dev/null 2>&1; then
  echo "'docker buildx' plugin not found — the build would fail on this Dockerfile's" >&2
  echo "BuildKit cache mounts. Install it, no root needed:" >&2
  echo >&2
  echo "  mkdir -p ~/.docker/cli-plugins" >&2
  echo "  BUILDX_VERSION=\$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep -oP '\"tag_name\": *\"\K[^\"]+')" >&2
  echo "  curl -sL \"https://github.com/docker/buildx/releases/download/\$BUILDX_VERSION/buildx-\$BUILDX_VERSION.linux-amd64\" -o ~/.docker/cli-plugins/docker-buildx" >&2
  echo "  chmod +x ~/.docker/cli-plugins/docker-buildx" >&2
  echo >&2
  echo "Or, with root: sudo apt-get install docker-buildx-plugin" >&2
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

# Reclaim what this rebuild just orphaned.
#
# The app image is ~18GB (CUDA + torch), so every rebuild that invalidates a late
# layer leaves the entire previous image behind, untagged and unreferenced. Nothing
# prunes that automatically, so the cost is linear in rebuild count: this filled a
# 503GB docker disk to 100% in about ten days — 33 dangling ~17.7GB images — which
# then broke writes inside *every* container on the host, not just this stack's.
#
# Two separate reclaims, because they are two separate stores and pruning images
# alone does not give the disk back — the space resurfaces as build cache:
#   docker image prune    dangling images, host-wide (nothing references them)
#   docker builder prune  BuildKit's cache, which is where the bulk actually sits
#
# `--keep-storage` is the point of doing this here rather than in a host-wide cron:
# it keeps the most recent cache so the *next* rebuild is still incremental, and
# only discards the older generations that no longer speed anything up.
#
# Runs after the stack is up, so a failed build never has its cache pruned out from
# under a retry, and is non-fatal: a rebuild that worked must not report failure
# because a cleanup did not.
if [ "${SKIP_PRUNE:-0}" != "1" ]; then
  echo
  echo "Reclaiming orphaned build artifacts (SKIP_PRUNE=1 to skip)..."
  docker image prune -f || echo "  (image prune failed; continuing)" >&2
  docker builder prune -f --keep-storage 20GB || echo "  (builder prune failed; continuing)" >&2
  echo "Docker disk now:"
  docker system df || true
fi

echo
docker compose ps
