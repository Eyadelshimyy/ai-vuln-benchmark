"""Idempotent tenant bootstrap for a fresh deployment.

Creates or updates the business identity and publishes its industry template
set. Re-running it is safe: the business row is matched by the identifier the
operator supplies and the seed re-publishes the same immutable version.

Identifiers come from arguments or the environment so no tenant UUID, channel
identifier or credential is checked in. Only non-secret identifiers are printed.

    gvas-bootstrap --business-id <uuid> --slug protech --name "ProTech" \
        --industry environmental_testing
"""

import argparse
import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.templates import PublishTemplateSetService
from gvas.config import Settings
from gvas.domain.identifiers import BusinessId
from gvas.domain.templates import IndustryKey, TemplateSetRef
from gvas.infrastructure.db import create_engine, create_session_factory
from gvas.infrastructure.industry_seeds import load_industry_definition
from gvas.infrastructure.unit_of_work import (
    SqlCompletenessUnitOfWorkFactory,
    SqlUnitOfWorkFactory,
)

DEFAULT_INDUSTRY = "environmental_testing"


class BootstrapInputError(ValueError):
    """The supplied tenant identity is missing or malformed."""


@dataclass(frozen=True)
class BootstrapRequest:
    business_id: BusinessId
    slug: str
    name: str
    industry: IndustryKey


@dataclass(frozen=True)
class BootstrapResult:
    business_id: BusinessId
    slug: str
    template_set: TemplateSetRef


def _environ(variable: str) -> str:
    value = os.environ.get(variable)
    return value if value is not None else ""


def _required(value: str | None, argument: str, variable: str) -> str:
    resolved = (value or _environ(variable)).strip()
    if not resolved:
        raise BootstrapInputError(f"{argument} is required (or set {variable})")
    return resolved


def build_request(arguments: argparse.Namespace) -> BootstrapRequest:
    raw_id = _required(arguments.business_id, "--business-id", "GVAS_BOOTSTRAP_BUSINESS_ID")
    try:
        business_id = BusinessId(UUID(raw_id))
    except ValueError as error:
        raise BootstrapInputError("--business-id must be a UUID") from error
    return BootstrapRequest(
        business_id=business_id,
        slug=_required(arguments.slug, "--slug", "GVAS_BOOTSTRAP_SLUG"),
        name=_required(arguments.name, "--name", "GVAS_BOOTSTRAP_NAME"),
        industry=IndustryKey(
            (arguments.industry or _environ("GVAS_BOOTSTRAP_INDUSTRY") or DEFAULT_INDUSTRY).strip()
        ),
    )


async def run_bootstrap(
    request: BootstrapRequest, session_factory: async_sessionmaker[AsyncSession]
) -> BootstrapResult:
    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        await unit_of_work.businesses.ensure(
            request.business_id,
            request.slug,
            request.name,
            now=datetime.now(UTC),
        )
        await unit_of_work.commit()
    publisher = PublishTemplateSetService(SqlCompletenessUnitOfWorkFactory(session_factory))
    template_set = await publisher.seed_industry(
        request.business_id, load_industry_definition(request.industry)
    )
    return BootstrapResult(
        business_id=request.business_id,
        slug=request.slug,
        template_set=template_set,
    )


async def bootstrap(request: BootstrapRequest) -> BootstrapResult:
    engine = create_engine(Settings().database_url)
    try:
        return await run_bootstrap(request, create_session_factory(engine))
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the initial GVAS business")
    parser.add_argument("--business-id")
    parser.add_argument("--slug")
    parser.add_argument("--name")
    parser.add_argument("--industry")
    result = asyncio.run(bootstrap(build_request(parser.parse_args(argv))))
    print(  # noqa: T201
        f"business {result.business_id} slug {result.slug} "
        f"template set {result.template_set.template_set_key} "
        f"version {result.template_set.version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
