#!/usr/bin/env bash
set -e

# Export host user permissions for build context injection
export APPUSER_UID=$(id -u)
export APPUSER_GID=$(id -g)

if [ "$1" = "dev" ]; then
  echo "⚠️  Targeting: Development Workspaces"
  ACTIVE_PROFILE="dev"
elif [ "$1" = "dev-combined" ]; then
  echo "⚠️  Targeting: Combined Development Workspace (inference + training, one container)"
  ACTIVE_PROFILE="dev-combined"
elif [ "$1" = "combined" ]; then
  echo "🚀 Targeting: Combined Production Server (inference + training, one container)"
  ACTIVE_PROFILE="combined"
else
  echo "🚀 Targeting: Production Core Server"
  ACTIVE_PROFILE="prod"
fi

echo "Stopping ALL running profiles to clear host ports..."
# This tears down every profile to prevent port collisions (they all share host ports)
docker compose --profile prod --profile dev --profile dev-combined --profile combined down --volumes --remove-orphans

echo "Rebuilding target layer contexts for profile: [$ACTIVE_PROFILE]..."
docker compose --profile $ACTIVE_PROFILE build \
  --build-arg APPUSER_UID=$APPUSER_UID \
  --build-arg APPUSER_GID=$APPUSER_GID

echo "Spinning up environment..."
docker compose --profile $ACTIVE_PROFILE up