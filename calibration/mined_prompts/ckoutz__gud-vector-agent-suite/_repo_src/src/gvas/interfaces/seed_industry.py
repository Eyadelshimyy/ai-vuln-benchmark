"""Load command for checked-in industry template seeds.

Onboarding copies industry rows into the business (copy-on-onboard), so this
command is idempotent: re-running it re-publishes the same immutable version and
leaves the business's own template-set assignment alone.

    python -m gvas.interfaces.seed_industry --business-id <uuid> --industry roofing
"""

import argparse
import asyncio
from collections.abc import Sequence
from uuid import UUID

from gvas.application.templates import PublishTemplateSetService
from gvas.config import Settings
from gvas.domain.identifiers import BusinessId
from gvas.domain.templates import IndustryKey, TemplateSetRef
from gvas.infrastructure.db import create_engine, create_session_factory
from gvas.infrastructure.industry_seeds import load_industry_definition
from gvas.infrastructure.unit_of_work import SqlCompletenessUnitOfWorkFactory


async def seed_industry(business_id: BusinessId, industry_key: IndustryKey) -> TemplateSetRef:
    engine = create_engine(Settings().database_url)
    try:
        publisher = PublishTemplateSetService(
            SqlCompletenessUnitOfWorkFactory(create_session_factory(engine))
        )
        return await publisher.seed_industry(business_id, load_industry_definition(industry_key))
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed a business with an industry template set")
    parser.add_argument("--business-id", required=True)
    parser.add_argument("--industry", required=True)
    arguments = parser.parse_args(argv)
    ref = asyncio.run(
        seed_industry(BusinessId(UUID(arguments.business_id)), IndustryKey(arguments.industry))
    )
    print(f"seeded {ref.template_set_key} version {ref.version}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
