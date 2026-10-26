"""Observe document-service behavior for a set of equivalence cases.

Run from inside ``services/document-service`` with that service's virtualenv so
the application modules import:

    poetry run python ../../security/equivalence/harness/emit_document_service.py \
        --cases ../../security/equivalence/cases/OW-SEC-401.json \
        --seed ../../security/equivalence/seed/document-service.json \
        --out /tmp/observed.json

The emitter only *observes*: it drives each case against a fixture built from
the seed alone and writes the raw outcome. It never decides whether a case
passed — ``equivalence_check.py`` grades the observations against the recording.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID

HARNESS_VERSION = 1

# The emitter is executed from the module directory so the application package
# imports; sys.path[0] is the harness directory, not the working directory.
sys.path.insert(0, os.getcwd())


def store_uuids_as_postgres_renders_them() -> None:
    """Make SQLite hold uuids hyphenated, the way Postgres renders one as text.

    SQLite has no uuid type, so SQLAlchemy stores bare hex there. The SQL under
    test compares a caller-supplied uuid *as text*, which would then never match
    on SQLite and would quietly reduce every owner-scoped case to an empty list -
    a fixture that records nothing while looking green. Production runs on
    Postgres, where the comparison matches, so the fixture stores the hyphenated
    form and both the ORM and the raw SQL agree with production.
    """
    from sqlalchemy import Uuid

    def bind_processor(self, dialect):  # noqa: ANN001, ANN202 - SQLAlchemy hook
        def process(value):  # noqa: ANN001, ANN202
            return None if value is None else str(value)

        return process

    Uuid.bind_processor = bind_processor


# The fixture lives in a fresh temporary directory every run, and paths leak into
# observable behaviour (a FileNotFoundError message names the file). Recording the
# raw path would make every rerun differ, so the fixture root is redacted to a
# stable token - the *shape* of the error is the contract, not the tmpdir name.
REDACTIONS: list[tuple[str, str]] = []


def redact(text: str) -> str:
    for needle, replacement in REDACTIONS:
        text = text.replace(needle, replacement)
    return text


def json_ready(value: Any) -> Any:
    """Return ``value`` reduced to JSON-safe primitives, order preserved."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return redact(str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


class Fixture:
    """The seeded document-service under test: database, archive, environment."""

    def __init__(self, seed: dict[str, Any], workdir: Path):
        self.seed = seed
        self.workdir = workdir
        self.archive_dir = workdir / "archive"
        self.outside_dir = workdir / "outside"

    def materialise_files(self) -> None:
        for name, body in self.seed.get("export_archive", {}).items():
            target = self.archive_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        for name, body in self.seed.get("outside_archive", {}).items():
            target = self.outside_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        os.environ["EXPORT_ARCHIVE_DIR"] = str(self.archive_dir)

    async def build_db(self) -> None:
        from app.db.base import Base
        from app.models.document import Document  # also registers the tables on Base
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        store_uuids_as_postgres_renders_them()
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with self.session_factory() as session:
            for row in self.seed.get("documents", []):
                session.add(Document(**self._document_kwargs(row)))
            await session.commit()

    @staticmethod
    def _document_kwargs(row: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(row)
        for key in ("id", "owner_id", "folder_id"):
            if kwargs.get(key):
                kwargs[key] = UUID(kwargs[key])
        for key in ("created_at", "updated_at"):
            kwargs[key] = datetime.fromisoformat(kwargs[key])
        return kwargs


async def call_case(fixture: Fixture, case: dict[str, Any]) -> Any:
    """Invoke a method on the subject class directly."""
    from app.services.document_query_repository import DocumentQueryRepository
    from app.services.export_archive import ExportArchive
    from app.services.share_link import ShareLinkService

    target = case["target"]
    args = case.get("args", {})
    if target == "document_query_repository":
        async with fixture.session_factory() as session:
            subject = DocumentQueryRepository(session)
            return await getattr(subject, case["method"])(**args)
    if target == "export_archive":
        return getattr(ExportArchive(), case["method"])(**args)
    if target == "share_link":
        return getattr(ShareLinkService(), case["method"])(**args)
    raise KeyError(f"unknown target: {target}")


class AppClient:
    """An httpx client bound to the application with the seeded session injected."""

    def __init__(self, fixture: Fixture):
        self.fixture = fixture

    async def __aenter__(self):
        from app.db.session import get_db
        from app.main import app
        from httpx import ASGITransport, AsyncClient

        self._app = app
        self._session_ctx = self.fixture.session_factory()
        session = await self._session_ctx.__aenter__()

        async def _override_get_db():
            yield session

        app.dependency_overrides[get_db] = _override_get_db
        self._client_ctx = AsyncClient(
            transport=ASGITransport(app=app), base_url="http://fixture"
        )
        return await self._client_ctx.__aenter__()

    async def __aexit__(self, *exc_info):
        await self._client_ctx.__aexit__(*exc_info)
        self._app.dependency_overrides.clear()
        await self._session_ctx.__aexit__(*exc_info)
        return False


def response_observation(response: Any) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return {
        "status": response.status_code,
        "content_type": response.headers.get("content-type"),
        "body": body,
    }


async def http_case(fixture: Fixture, case: dict[str, Any]) -> Any:
    """Drive a request through the FastAPI application."""
    request = case["request"]
    async with AppClient(fixture) as client:
        response = await client.request(
            request.get("method", "GET"),
            request["path"],
            params=request.get("params"),
            json=request.get("json"),
            headers=request.get("headers"),
        )
    return response_observation(response)


async def share_link_http_roundtrip(fixture: Fixture, args: dict[str, Any]) -> dict[str, Any]:
    """Mint a share link over HTTP and read the document back through it."""
    document_id = args["document_id"]
    owner_id = args["owner_id"]
    headers = {"Authorization": "Bearer fixture", "X-User-ID": owner_id}
    async with AppClient(fixture) as client:
        minted = await client.post(
            f"/api/v1/documents/{document_id}/share", headers=headers
        )
        token = minted.json().get("token") if minted.status_code == 200 else None
        shared = await client.get(
            "/api/v1/documents/shared",
            params={"document_id": document_id, "token": token or "none"},
        )
        tampered = await client.get(
            "/api/v1/documents/shared",
            params={"document_id": document_id, "token": f"{token or 'none'}x"},
        )
    return {
        "mint_status": minted.status_code,
        "shared_status": shared.status_code,
        "shared_document_id": shared.json().get("id") if shared.status_code == 200 else None,
        "tampered_status": tampered.status_code,
    }


async def probe_case(fixture: Fixture, case: dict[str, Any]) -> Any:
    """Run a named attacker-side probe against the subject."""
    from app.services.share_link import ShareLinkService

    name = case["probe"]
    args = case.get("args", {})
    if name == "share_link_contract":
        service = ShareLinkService()
        document_id = args["document_id"]
        other_id = args["other_document_id"]
        token = service.mint_token(document_id)
        return {
            "minted_token_verifies": service.verify_token(document_id, token),
            "token_stable_across_calls": token == service.mint_token(document_id),
            "other_documents_token_rejected": not service.verify_token(
                document_id, service.mint_token(other_id)
            ),
            "garbage_token_rejected": not service.verify_token(document_id, "not-a-token"),
        }
    if name == "share_link_http_roundtrip":
        return await share_link_http_roundtrip(fixture, args)
    if name == "offline_share_token_forgery":
        # An attacker who reads the published source knows the algorithm and the
        # default salt; nothing secret is needed to derive a token.
        document_id = args["document_id"]
        guess = hashlib.md5(
            f"{document_id}:{args['salt']}".encode(), usedforsecurity=False
        ).hexdigest()[: args.get("length", 16)]
        return {
            "forged_token_accepted": ShareLinkService().verify_token(document_id, guess),
        }
    raise KeyError(f"unknown probe: {name}")


KINDS = {"call": call_case, "http": http_case, "probe": probe_case}


async def observe(fixture: Fixture, case: dict[str, Any]) -> dict[str, Any]:
    handler = KINDS.get(case["kind"])
    if handler is None:
        raise KeyError(f"unknown case kind: {case['kind']}")
    try:
        value = await handler(fixture, case)
    except Exception as exc:  # noqa: BLE001 - the error itself is the observation
        return {
            "id": case["id"],
            "outcome": "error",
            "error_type": type(exc).__name__,
            "error_message": redact(str(exc)),
        }
    return {"id": case["id"], "outcome": "ok", "value": json_ready(value)}


def interface_modules() -> dict[str, ModuleType]:
    """The modules whose signatures a finding may pin, keyed by dotted name.

    An allowlist rather than an import of whatever the case file names: the case
    file is data, and importing an arbitrary module named in data would execute
    it.
    """
    from app.services import document_query_repository, export_archive, share_link

    return {
        "app.services.document_query_repository": document_query_repository,
        "app.services.export_archive": export_archive,
        "app.services.share_link": share_link,
    }


def capture_interface(members: list[str]) -> dict[str, str]:
    """Record the signature of every public member the finding must preserve."""
    modules = interface_modules()
    signatures: dict[str, str] = {}
    for member in members:
        module_name, _, attribute = member.partition(":")
        module = modules.get(module_name)
        if module is None:
            raise KeyError(f"module is not an interface subject: {module_name}")
        obj: object = module
        for part in attribute.split("."):
            obj = getattr(obj, part)
        signatures[member] = f"{attribute.split('.')[-1]}{inspect.signature(obj)}"
    return signatures


async def main() -> int:
    args = parse_args()
    spec = json.loads(args.cases.read_text(encoding="utf-8"))
    seed = json.loads(args.seed.read_text(encoding="utf-8"))

    for key, value in spec.get("env", {}).items():
        os.environ[key] = value

    with tempfile.TemporaryDirectory(prefix="ow-equivalence-") as tmp:
        fixture = Fixture(seed, Path(tmp))
        REDACTIONS.append((str(fixture.workdir), "<fixture>"))
        fixture.materialise_files()
        await fixture.build_db()
        observations = [await observe(fixture, case) for case in spec["cases"]]
        await fixture.engine.dispose()

    payload = {
        "finding": spec["finding"],
        "module": spec["module"],
        "harness_version": HARNESS_VERSION,
        "interface": capture_interface(spec.get("interface", [])),
        "cases": observations,
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
