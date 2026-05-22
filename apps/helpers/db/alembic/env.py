import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add project root to sys.path so 'apps' package is importable regardless of CWD
_project_root = Path(__file__).parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sqlmodel import SQLModel

# Import all models so Alembic autogenerate can detect the full schema
from apps.helpers.db.types import (  # noqa: F401
    BBox,
    BBoxTagLink,
    ImageLabel,
    ImageRecord,
    Label,
    Source,
    Tag,
    TaskConfig,
    TaskInstance,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Allow CLI callers to pass DB path via: alembic -x db_url=sqlite:///path/to/db
    x_db_url = (context.get_x_argument(as_dictionary=True) or {}).get("db_url")
    url = x_db_url or config.get_main_option("sqlalchemy.url")
    if not url:
        # No URL configured — use a fresh temp DB so autogenerate produces
        # CREATE TABLE statements for all models in target_metadata.
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        tmp.close()
        url = f"sqlite:///{tmp.name}"
    config.set_main_option("sqlalchemy.url", url)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
