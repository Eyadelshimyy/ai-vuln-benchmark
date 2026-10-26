"""Shared fixtures: real manifest + copybooks, in-memory drivers, a small SEED-SPEC-shaped dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from ldm.context import Log, RunContext
from ldm.convert import Value
from ldm.drivers.fakes import FakeSource, FakeTarget
from ldm.runner import build_context
from ldm.staging import DirectoryBlobStore

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "migration" / "manifest.yaml"
CUTOFF = "2019-01-01-00.00.00.000000000000"
TS0 = "2010-01-01-00.00.00.000000000000"

POLICIES = [
    ("FIN7", "Financial records", 7, "", "Y"),
    ("LGL7", "Legal correspondence", 7, "", "Y"),
    ("HRS7", "Personnel files", 7, "", "Y"),
    ("TAX7", "Tax filings", 7, "", "Y"),
    ("AUD7", "Internal audit workpapers", 7, "", "Y"),
    ("F07R", "Financial records (retired code)", 7, "FIN7", "N"),
    ("L07R", "Legal correspondence (retired code)", 7, "LGL7", "N"),
    ("H07R", "Personnel files (retired code)", 7, "HRS7", "N"),
    ("OPS1", "Operations records", 1, "", "Y"),
    ("PERM", "Permanent records", 99, "", "Y"),
]


def cp037(text: str, length: int) -> bytes:
    return text.encode("cp037").ljust(length, b"\x40")


def docarch_row(
    key: str,
    cls: str,
    last_access: str,
    *,
    charge: Decimal | None = None,
    unit_rate: Decimal = Decimal("0.00001000"),
    owner: bytes | None = None,
    disp_dt: bytes | None = None,
    byte_size: int = 4096,
) -> dict[str, Value]:
    padded = key.ljust(16)
    return {
        "ARCH_KEY": padded,
        "DOC_ID": f"{hashlib.md5(key.encode()).hexdigest()[:8]}-0000-4000-8000-000000000000",  # noqa: S324
        "VERSION_NO": 1,
        "RETENTION_CLASS": cls.ljust(4),
        "LAST_ACCESS_TS": last_access,
        "STORAGE_CHARGE": charge if charge is not None else (Decimal(byte_size) * unit_rate),
        "UNIT_RATE": unit_rate,
        "OWNER_NAME": owner if owner is not None else cp037("LOPEZ, M.", 40),
        "DISPOSITION_DT": disp_dt if disp_dt is not None else cp037("2023" + last_access[5:7] + last_access[8:10], 8),
        "LEGAL_HOLD_FLAG": "N",
        "CHECKSUM_ALG": "SHA256  ",
        "CONTENT_SHA256": hashlib.sha256(padded.encode("ascii")).hexdigest(),
        "BYTE_SIZE": byte_size,
        "SOURCE_SYS": "OWD",
    }


def fileaud_row(key: str, arch_key: str, cls: str, event_ts: str) -> dict[str, Value]:
    return {
        "AUDIT_KEY": key.ljust(20),
        "ARCH_KEY": arch_key.ljust(16),
        "EVENT_TYPE": "VIEW",
        "EVENT_TS": event_ts,
        "ACTOR_ID": "U00000000001",
        "RETENTION_CLASS": cls.ljust(4),
        "DISPOSITION_CODE": "00",
        "CLIENT_IP": "10.1.2.3".ljust(15),
        "DETAIL_TEXT": "VIEW v1".ljust(40),
    }


def policy_row(code: str, desc: str, years: int, successor: str, active: str) -> dict[str, Value]:
    return {
        "POLICY_CODE": code.ljust(4),
        "POLICY_DESC": desc.ljust(60),
        "RETENTION_YEARS": years,
        "SUCCESSOR_CODE": successor.ljust(4),
        "ACTIVE_FLAG": active,
        "DISPOSITION_ACTION": ("PERM" if code == "PERM" else "DEST").ljust(4),
        "EFFECTIVE_TS": TS0,
    }


@dataclass
class Seed:
    """Expected outcome of the seeded dataset (mirrors SEED-SPEC §3 at small scale)."""

    source: FakeSource
    docarch_selected: int = 0
    fileaud_selected: int = 0
    planted: dict[str, list[str]] = field(default_factory=dict)  # MIG class -> padded source keys


def seed_source(*, generated: int = 30, children_per_parent: int = 2, plant: bool = True) -> Seed:
    src = FakeSource()
    seed = Seed(source=src)
    src.add_rows("ARCHIVE", "RETNPLCY", [policy_row(*p) for p in POLICIES])
    classes = ["FIN7", "LGL7", "HRS7", "TAX7", "AUD7"]
    docs: list[dict[str, Value]] = []
    auds: list[dict[str, Value]] = []
    for g in range(generated):
        key = f"DA{g + 1:014d}"
        selected = g % 3 != 2
        cls = classes[g % 5] if g % 7 != 6 else "OPS1"
        old = f"201{5 + g % 4}-0{1 + g % 9}-1{g % 9}-08.30.00.{g:012d}"
        ts = old if selected else f"2022-01-0{1 + g % 9}-00.00.00.000000000000"
        docs.append(docarch_row(key, cls, ts))
        is_selected = selected and cls != "OPS1"
        if is_selected:
            seed.docarch_selected += 1
        for c in range(children_per_parent):
            akey = f"FA{g * 10 + c + 1:018d}"
            child_ts = ts if is_selected else f"2022-06-0{1 + c}-00.00.00.000000000000"
            auds.append(fileaud_row(akey, key, cls, child_ts))
            if is_selected:
                seed.fileaud_selected += 1
    if plant:
        p = seed.planted
        p["MIG-01"] = [f"MIG01-{k:010d}".ljust(16) for k in range(1, 3)]
        for k, key in enumerate(p["MIG-01"], start=1):
            docs.append(
                docarch_row(
                    key.strip(),
                    "FIN7",
                    f"2016-03-0{k}-10.15.30.123456789012",
                    owner=cp037("LOPEZ", 5) + b"\x3f" + cp037(", M.", 34),
                )
            )
        p["MIG-02"] = [f"MIG02-{k:010d}".ljust(16) for k in range(1, 3)]
        for k, key in enumerate(p["MIG-02"], start=1):
            docs.append(
                docarch_row(
                    key.strip(),
                    "TAX7",
                    f"2015-06-1{k}-08.00.00.000000000001",
                    unit_rate=Decimal(f"12345678901234.5678900{k}"),
                    charge=Decimal("1.00000000"),
                )
            )
        p["MIG-03"] = [f"MIG03-{k:010d}".ljust(16) for k in range(1, 3)]
        for k, key in enumerate(p["MIG-03"], start=1):
            docs.append(docarch_row(key.strip(), "LGL7", f"2014-11-2{k}-17.45.00.500000000000", disp_dt=b"\x00" * 8))
        p["MIG-04"] = [f"MIG04-0{k}".ljust(16) for k in range(1, 3)]
        for k, key in enumerate(p["MIG-04"], start=1):
            docs.append(docarch_row(key.strip(), "HRS7", f"2013-02-0{k}-09.30.00.000000000000"))
        # MIG-05: parents not selected (recent), children selected (old) -> orphans
        parents = [f"MIG05-{k:010d}" for k in range(1, 3)]
        for k, key in enumerate(parents, start=1):
            docs.append(docarch_row(key, "FIN7", f"2025-02-0{k}-12.00.00.000000000000"))
        p["MIG-05"] = [f"MIG05-{k:014d}".ljust(20) for k in range(1, 3)]
        for k, key in enumerate(p["MIG-05"], start=1):
            auds.append(fileaud_row(key.strip(), parents[k - 1], "FIN7", f"2017-05-0{k}-07.00.00.000000000000"))
        p["MIG-06"] = [f"MIG06-{k:010d}".ljust(16) for k in range(1, 3)]
        for k, key in enumerate(p["MIG-06"], start=1):
            docs.append(docarch_row(key.strip(), "AUD7", f"2016-09-0{k}-11.11.11.111111111111"))
        p["MIG-07"] = [f"MIG07-{k:010d}".ljust(16) for k in range(1, 13)]
        f07 = [Decimal(v) for v in ("100", "200", "300", "400", "500", "1100.12345678")]
        l07 = [Decimal(v) for v in ("433", "433", "433", "433", "433", "435.12345679")]
        for k, key in enumerate(p["MIG-07"], start=1):
            cls, charge = ("F07R", f07[k - 1]) if k <= 6 else ("L07R", l07[k - 7])
            ts = f"2017-0{1 if k <= 6 else 2}-0{(k - 1) % 6 + 1}-00.00.00.000000000000"
            docs.append(docarch_row(key.strip(), cls, ts, charge=charge))
        seed.docarch_selected += sum(len(p[c]) for c in ("MIG-01", "MIG-02", "MIG-03", "MIG-04", "MIG-06", "MIG-07"))
        seed.fileaud_selected += len(p["MIG-05"])
    src.add_rows("ARCHIVE", "DOCARCH", docs)
    src.add_rows("ARCHIVE", "FILEAUD", auds)
    return seed


def make_ctx(
    tmp_path: Path,
    source: FakeSource,
    target: FakeTarget | None = None,
    *,
    namespace: str = "t01-after",
    run_id: str = "run-1",
    manifest: Path = MANIFEST,
    env: dict[str, str] | None = None,
) -> RunContext:
    base_env = {
        "LOCAL_STAGING_DIR": str(tmp_path / "staging"),
        "LDM_UNLOAD_MODE": "builtin",
        "LDM_HOST": "local",
    }
    base_env.update(env or {})
    return build_context(
        manifest,
        namespace,
        run_id,
        env=base_env,
        source=source,
        target=target or FakeTarget(),
        blobs=DirectoryBlobStore(tmp_path / "blobs"),
        log=Log(),
    )


AZURE_TARGET_OVERLAY = """target:
  provider: azuresql
  connection_env:
    server: AZSQL_SERVER
    database: AZSQL_DATABASE
    user: AZSQL_USER
    password: AZSQL_PASSWORD
    auth_mode: AZSQL_AUTH
    managed_identity_client_id: AZURE_CLIENT_ID
  ddl_dir: migration/target/sql
  typemap: migration/job/typemaps/db2-to-azuresql.yaml
"""


def make_manifest_tree(
    tmp_path: Path,
    token: str,
    *,
    purge: bool = True,
    before: bool = False,
    azure_target: bool = False,
    load_engine: str = "serial",
) -> Path:
    """Copy the repo manifest into <tmp>/<token>/migration with run_token=<token> and before/after overlays.

    Relative manifest paths (copybooks, typemaps, DDL) resolve through symlinks to the real repo tree.
    Returns the base manifest path.
    """
    d = tmp_path / token / "migration"
    (d / "manifests").mkdir(parents=True)
    for sub in ("source", "job", "target", "sessions"):
        (d / sub).symlink_to(REPO_ROOT / "migration" / sub, target_is_directory=True)
    base = MANIFEST.read_text(encoding="utf-8").replace("run_token: d24", f"run_token: {token}")
    (d / "manifest.yaml").write_text(base, encoding="utf-8")
    (d / "manifests" / f"{token}-after.yaml").write_text(
        f"namespace: {token}-after\nextends: ../manifest.yaml\nmigrate: true\n"
        f"azure: {'true' if azure_target else 'false'}\npurge: {'true' if purge else 'false'}\n"
        f"execution:\n  load_engine: {load_engine}\n" + (AZURE_TARGET_OVERLAY if azure_target else ""),
        encoding="utf-8",
    )
    (d / "manifests" / f"{token}-before.yaml").write_text(
        f"namespace: {token}-before\nextends: ../manifest.yaml\nmigrate: false\nazure: false\npurge: false\n",
        encoding="utf-8",
    )
    return d / "manifest.yaml"


@pytest.fixture
def manifest_after(tmp_path: Path) -> Path:
    return make_manifest_tree(tmp_path, "t01", purge=True)


@pytest.fixture
def manifest_dry(tmp_path: Path) -> Path:
    return make_manifest_tree(tmp_path, "t02", purge=False)
