import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Import every mapped class before target_metadata is evaluated so
# ``alembic revision --autogenerate`` can detect HackerOne and agent changes.
from app.models.base import Base  # noqa: F401
import app.models  # noqa: F401
import app.models.analysis  # noqa: F401
import app.models.debugging  # noqa: F401
import app.models.discovery  # noqa: F401
import app.models.finding  # noqa: F401
import app.models.github  # noqa: F401
import app.models.hackerone  # noqa: F401
import app.models.project  # noqa: F401
import app.models.repair  # noqa: F401
import app.models.reproduction  # noqa: F401
import app.models.security_agent  # noqa: F401
import app.models.security_audit  # noqa: F401
import app.models.security_finding  # noqa: F401
import app.models.test_generation  # noqa: F401
import app.models.test_run  # noqa: F401
import app.models.verification  # noqa: F401

config = context.config
fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    return config.get_main_option("sqlalchemy.url", "")


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(get_url(), poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
