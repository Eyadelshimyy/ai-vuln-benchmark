#!/usr/bin/env python
"""Build a currency-explicit enterprise due-diligence gate manifest.

The gate is evidence-oriented and amount-neutral by name. A monetary scenario
may be attached for procurement review, but the resulting manifest always
states that it is not a valuation claim.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence


CANONICAL_GATE_NAME = "enterprise_due_diligence_gate"
DEFAULT_CURRENCY_CODE = "KRW"
DEFAULT_SCENARIO_AMOUNT = 2_000_000_000
SCHEMA_VERSION = "1.0.0"
LEGACY_GATE_ALIASES = frozenset(
    {
        "20b",
        "20b_product",
        "20b_product_readiness",
        "require_20b_product",
    }
)
_GIT_HEX_DIGITS = frozenset("0123456789abcdef")


def _safe_output_path(output_path: Path) -> Path:
    """Resolve one output path below the current directory without symlinks."""

    root = Path.cwd().resolve()
    candidate = Path(output_path)
    if candidate.is_absolute():
        raise ValueError("output path must be relative to the current working directory")
    if ".." in candidate.parts:
        raise ValueError("output path must remain within the current working directory")

    probe = root
    for part in candidate.parts:
        probe /= part
        if probe.is_symlink():
            raise ValueError("output path must not contain symbolic links")

    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("output path must remain within the current working directory") from None
    if resolved.exists() and not resolved.is_file():
        raise ValueError("output path must name a regular file")
    return resolved


def normalize_gate_name(value: str) -> str:
    """Return the canonical gate name and warn for a supported legacy alias."""

    if type(value) is not str:
        raise ValueError("gate_name must be an exact built-in string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized == CANONICAL_GATE_NAME:
        return CANONICAL_GATE_NAME
    if normalized in LEGACY_GATE_ALIASES:
        warnings.warn(
            (
                f"{value!r} is deprecated; use {CANONICAL_GATE_NAME!r}. "
                "The legacy alias describes neither a currency nor a valuation contract."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return CANONICAL_GATE_NAME
    raise ValueError(f"unsupported due-diligence gate name: {value!r}")


def validate_currency_code(value: str) -> str:
    """Validate and normalize an ISO-4217-style three-letter currency code."""

    if type(value) is not str:
        raise ValueError("currency_code must be an exact built-in string")
    normalized = value.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("currency_code must be exactly three ASCII letters")
    return normalized


def validate_scenario_amount(value: int) -> int:
    """Validate a positive procurement scenario amount without accepting subclasses."""

    if type(value) is not int or value <= 0:
        raise ValueError("scenario_amount must be a positive integer")
    return value


def validate_source_commit(value: str) -> str:
    """Validate a callback-free canonical full Git object identity."""

    if (
        type(value) is not str
        or len(value) not in (40, 64)
        or any(character not in _GIT_HEX_DIGITS for character in value)
    ):
        raise ValueError("source_commit must be a canonical lowercase full Git object identity")
    return value


def build_gate_manifest(
    *,
    source_commit: str,
    gate_name: str = CANONICAL_GATE_NAME,
    currency_code: str = DEFAULT_CURRENCY_CODE,
    scenario_amount: int = DEFAULT_SCENARIO_AMOUNT,
    valuation_claim: bool = False,
) -> dict[str, Any]:
    """Build the deterministic public contract for an enterprise evidence gate."""

    if not isinstance(valuation_claim, bool):
        raise ValueError("valuation_claim must be a boolean")
    if valuation_claim:
        raise ValueError("enterprise due-diligence evidence must not be a valuation claim")

    canonical_gate_name = normalize_gate_name(gate_name)
    normalized_currency = validate_currency_code(currency_code)
    normalized_amount = validate_scenario_amount(scenario_amount)
    normalized_commit = validate_source_commit(source_commit)
    scenario_name = f"{normalized_currency.lower()}_{normalized_amount}_procurement_scenario"

    return {
        "currency_code": normalized_currency,
        "gate_name": canonical_gate_name,
        "legacy_gate_aliases": sorted(LEGACY_GATE_ALIASES),
        "scenario_amount": normalized_amount,
        "scenario_name": scenario_name,
        "schema_version": SCHEMA_VERSION,
        "source_commit": normalized_commit,
        "valuation_claim": False,
    }


def _portable_existing_mode(validated_path: Path) -> int | None:
    """Return an existing regular target's permissions for atomic replacement."""
    try:
        target_stat = validated_path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(target_stat.st_mode):
        raise ValueError("output path must name a regular file")
    return stat.S_IMODE(target_stat.st_mode)


def _write_manifest_descriptor(
    output_path: Path,
    content: str,
    validated_path: Path,
) -> None:
    """Write content with descriptor safety or an atomic portable fallback."""
    if (
        os.name != "posix"
        or os.open not in os.supports_dir_fd
        or os.mkdir not in os.supports_dir_fd
        or os.rename not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or not hasattr(os, "fchmod")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        validated_path.parent.mkdir(parents=True, exist_ok=True)
        existing_mode = _portable_existing_mode(validated_path)
        temporary_path: Path | None = None
        temporary_fd: int | None = None
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            for _ in range(128):
                candidate = validated_path.parent / (
                    f".{validated_path.name}.{secrets.token_hex(16)}.tmp"
                )
                try:
                    temporary_fd = os.open(candidate, temporary_flags, 0o666)
                except FileExistsError:
                    continue
                temporary_path = candidate
                break
            else:
                raise ValueError("manifest output could not be written")

            assert temporary_fd is not None
            try:
                stream = os.fdopen(temporary_fd, "w", encoding="utf-8")
            except BaseException:
                os.close(temporary_fd)
                temporary_fd = None
                raise
            temporary_fd = None
            with stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            assert temporary_path is not None
            if existing_mode is not None:
                os.chmod(temporary_path, existing_mode)
            os.replace(temporary_path, validated_path)
            temporary_path = None
        finally:
            if temporary_fd is not None:
                try:
                    os.close(temporary_fd)
                except OSError:
                    pass
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return

    components = output_path.parts
    if not components:
        raise ValueError("output path must name a regular file")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(".", directory_flags)
    try:
        for component in components[:-1]:
            try:
                os.mkdir(component, 0o755, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_directory_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_directory_fd

        try:
            target_stat = os.stat(
                components[-1],
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing_mode = None
        else:
            if not stat.S_ISREG(target_stat.st_mode):
                raise ValueError("output path must name a regular file")
            existing_mode = stat.S_IMODE(target_stat.st_mode)

        temporary_name: str | None = None
        temporary_fd: int | None = None
        temporary_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            for _ in range(128):
                candidate = f".{components[-1]}.{secrets.token_hex(16)}.tmp"
                try:
                    temporary_fd = os.open(
                        candidate,
                        temporary_flags,
                        0o666,
                        dir_fd=directory_fd,
                    )
                except FileExistsError:
                    continue
                temporary_name = candidate
                break
            else:
                raise ValueError("manifest output could not be written")

            assert temporary_fd is not None
            if existing_mode is not None:
                os.fchmod(temporary_fd, existing_mode)
            try:
                stream = os.fdopen(temporary_fd, "w", encoding="utf-8")
            except BaseException:
                os.close(temporary_fd)
                temporary_fd = None
                raise
            temporary_fd = None
            with stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

            assert temporary_name is not None
            os.rename(
                temporary_name,
                components[-1],
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_name = None
        finally:
            if temporary_fd is not None:
                try:
                    os.close(temporary_fd)
                except OSError:
                    pass
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def write_gate_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """Write deterministic UTF-8 JSON through a validated relative path."""

    validated_path = _safe_output_path(output_path)
    content = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        _write_manifest_descriptor(Path(output_path), content, validated_path)
    except ValueError:
        raise
    except (NotImplementedError, OSError):
        raise ValueError("manifest output could not be written") from None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the gate manifest utility."""

    parser = argparse.ArgumentParser(
        description="Build a currency-explicit enterprise due-diligence gate manifest."
    )
    parser.add_argument(
        "--gate-name",
        default=CANONICAL_GATE_NAME,
        help="Canonical gate name or a supported legacy alias during deprecation.",
    )
    parser.add_argument(
        "--currency-code",
        default=DEFAULT_CURRENCY_CODE,
        help="Three-letter procurement-scenario currency code.",
    )
    parser.add_argument(
        "--scenario-amount",
        type=int,
        default=DEFAULT_SCENARIO_AMOUNT,
        help="Positive integer procurement-scenario amount.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Canonical lowercase full SHA-1 or SHA-256 Git object identity.",
    )
    parser.add_argument(
        "--out",
        default="enterprise_due_diligence_gate.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--require-enterprise-due-diligence",
        action="store_true",
        help="Require the canonical evidence gate. Retained for orchestration symmetry.",
    )
    parser.add_argument(
        "--require-20b-product",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    selected_gate_name = args.gate_name
    if args.require_20b_product:
        warnings.warn(
            (
                "--require-20b-product is deprecated; use "
                "--require-enterprise-due-diligence."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        selected_gate_name = "require_20b_product"
    try:
        manifest = build_gate_manifest(
            source_commit=args.source_commit,
            gate_name=selected_gate_name,
            currency_code=args.currency_code,
            scenario_amount=args.scenario_amount,
            valuation_claim=False,
        )
        output_path = Path(args.out)
        write_gate_manifest(manifest, output_path)
    except ValueError as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}, sort_keys=True), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "gate_name": manifest["gate_name"],
                "out": str(output_path),
                "status": "ok",
                "valuation_claim": manifest["valuation_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
