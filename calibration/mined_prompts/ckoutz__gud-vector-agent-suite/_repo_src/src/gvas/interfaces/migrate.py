"""Release/pre-deploy migration entrypoint.

Runs Alembic in-process against the packaged migration directory, so the
deployment does not depend on the working directory or on a checked-out
``alembic.ini``. ``env.py`` reads the database URL from settings, which is where
the managed-provider URL is normalized.

    gvas-migrate [--revision head]
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "infrastructure" / "migrations"


def build_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    return config


def upgrade(revision: str = "head") -> None:
    command.upgrade(build_config(), revision)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply GVAS database migrations")
    parser.add_argument("--revision", default="head")
    arguments = parser.parse_args(argv)
    upgrade(arguments.revision)
    print(f"migrated to {arguments.revision}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
