"""per-source mqtt name on watch profile sources

Adds `watch_profile_sources.mqtt_name` — what `{source}` renders as in this watcher's
MQTT topics. NULL means "use the source's own name", so every existing watcher keeps
publishing to exactly the topics it publishes to today.

On the junction rather than on `sources` because the two names answer to different
audiences: a source's name is chosen to be recognisable when picking a camera out of a
dropdown, while this is a topic level someone's Home Assistant automation is written
against. Renaming the camera should not silently re-route its topics.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "737629e0e63c"
down_revision: Union[str, None] = "816e4042c730"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("mqtt_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.drop_column("mqtt_name")
