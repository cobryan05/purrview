#!/usr/bin/env bash
set -e

# Export host user permissions for build context injection
export APPUSER_UID=$(id -u)
export APPUSER_GID=$(id -g)

if [ "$1" = "dev" ]; then
  echo "⚠️  Targeting: Development Workspaces"
  ACTIVE_PROFILE="dev"
else
  echo "🚀 Targeting: Production Core Server"
  ACTIVE_PROFILE="prod"
fi

echo "Stopping ALL running profiles to clear host ports..."
# This tears down everything under both profiles to prevent port collisions
docker compose --profile prod --profile dev down --volumes --remove-orphans

echo "Rebuilding target layer contexts for profile: [$ACTIVE_PROFILE]..."
docker compose --profile $ACTIVE_PROFILE build \
  --build-arg APPUSER_UID=$APPUSER_UID \
  --build-arg APPUSER_GID=$APPUSER_GID

echo "Spinning up environment..."
docker compose --profile $ACTIVE_PROFILE up