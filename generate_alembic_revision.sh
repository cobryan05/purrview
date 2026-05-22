#!/usr/bin/env bash
# Usage: ./generate_alembic_revision.sh "describe your schema change" [/path/to/db.sqlite]
#
# Generates a new Alembic migration based on the current SQLModel schema.
# If no DB path is given, a temporary DB is created, upgraded to head, then
# used for autogenerate — so only schema deltas vs existing migrations appear.
# Pass the real DB path to detect only the delta against the live database.
# Run from the project root. Review the generated file before committing.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALEMBIC_INI="$SCRIPT_DIR/apps/helpers/db/alembic.ini"
MESSAGE="${1:-auto}"

if [ -n "$2" ]; then
    DB_URL="sqlite:///$2"
else
    DB_URL="sqlite:////data/db/db.sqlite"
fi

# Bring the DB to head before autogenerating so we only see the delta.
alembic -c "$ALEMBIC_INI" -x db_url="$DB_URL" upgrade head
alembic -c "$ALEMBIC_INI" -x db_url="$DB_URL" revision --autogenerate -m "$MESSAGE"
