"""add mqtt_name to interaction_rules and labels

Adds the Home Assistant name override a rule and a label publish under: NULL or blank
means "use the row's own name", so every existing row keeps the topics it has.

Revision ID: 12092cc491f4
Revises: ab612fbceb7d
Create Date: 2026-10-03 19:32:51.229885

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "12092cc491f4"
down_revision: Union[str, None] = "ab612fbceb7d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(sa.Column("mqtt_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.add_column(sa.Column("mqtt_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.drop_column("mqtt_name")

    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("mqtt_name")
