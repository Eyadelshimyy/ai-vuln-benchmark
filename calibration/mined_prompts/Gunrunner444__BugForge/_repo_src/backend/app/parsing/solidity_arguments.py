"""Bounded Solidity argument candidates.

A candidate is a literal to try, not a proof that the value is meaningful.
Unsupported types stay unsupported. Combinations are zipped, not multiplied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UINT = re.compile(r"^uint(\d*)$")
_INT = re.compile(r"^int(\d*)$")
_BYTES_N = re.compile(r"^bytes(\d+)$")
_ARRAY = re.compile(r"^(?P<inner>.+)\[(?P<length>\d*)\]$")


@dataclass(frozen=True)
class ArgumentCandidate:
    literal: str
    provenance: str
    type_name: str


@dataclass(frozen=True)
class DerivedArgument:
    literal: str
    observation: str
    transaction_index: int
    snapshot_id: str
    transformation: str
    assumption: str


@dataclass(frozen=True)
class StateObservation:
    snapshot_id: str
    transaction_index: int
    name: str
    value: str


_STATE_NAMES = frozenset(
    {
        "balance",
        "balances",
        "shares",
        "totalassets",
        "totalsupply",
        "reserve",
        "reserves",
        "debt",
        "allowance",
        "allowances",
        "nonce",
        "nonces",
    }
)


def candidates_for_type(type_name: str, *, limit: int = 4) -> tuple[ArgumentCandidate, ...] | None:
    """A small deterministic set, or None when the type cannot be represented."""
    bound = max(1, min(int(limit), 4))
    found = _candidates(type_name.strip())
    if found is None:
        return None
    return tuple(found[:bound])


def preferred_literal(type_name: str) -> str | None:
    """One replay literal. None means the type stays unsupported."""
    found = candidates_for_type(type_name)
    if not found:
        return None
    for item in found:
        if item.provenance == "one":
            return item.literal
    return found[0].literal


def combine(
    type_names: tuple[str, ...], *, limit: int = 4
) -> tuple[tuple[ArgumentCandidate, ...], ...] | None:
    """Zip candidates across parameters. This is not a Cartesian product."""
    columns: list[tuple[ArgumentCandidate, ...]] = []
    for name in type_names:
        found = candidates_for_type(name, limit=limit)
        if found is None:
            return None
        columns.append(found)
    if not columns:
        return ((),)
    width = min(limit, max(len(column) for column in columns))
    rows: list[tuple[ArgumentCandidate, ...]] = []
    for index in range(width):
        rows.append(tuple(column[min(index, len(column) - 1)] for column in columns))
    return tuple(rows)


def derive_from_state(observation: StateObservation) -> DerivedArgument | None:
    """A later argument may copy an observed value. The observation is not trusted."""
    if observation.name.lower() not in _STATE_NAMES:
        return None
    if not re.fullmatch(r"\d+", observation.value.strip()):
        return None
    return DerivedArgument(
        observation.value.strip(),
        observation.name,
        observation.transaction_index,
        observation.snapshot_id,
        "copy",
        "observed state is not trusted",
    )


def parameter_types(header: str) -> tuple[str, ...] | None:
    """Types from one function header. None when a type cannot be named."""
    if "(" not in header:
        return ()
    params = header[header.find("(") + 1 : header.rfind(")")] if ")" in header else ""
    if not params.strip():
        return ()
    found: list[str] = []
    for part in _split_args(params):
        tokens = [
            token
            for token in part.replace("memory", " ")
            .replace("calldata", " ")
            .replace("storage", " ")
            .split()
            if token not in {"payable", "indexed"}
        ]
        if not tokens:
            return None
        found.append(tokens[0])
    return tuple(found)


def _candidates(type_name: str) -> tuple[ArgumentCandidate, ...] | None:
    if not type_name or "mapping" in type_name or type_name.startswith("function"):
        return None
    array = _ARRAY.fullmatch(type_name.replace(" ", ""))
    if array:
        inner = array.group("inner")
        length = array.group("length")
        if length and int(length) > 1:
            return None
        inner_found = _candidates(inner)
        if inner_found is None:
            return None
        element = ArgumentCandidate(f"[{inner_found[0].literal}]", "boundary", type_name)
        if length == "1":
            return (element,)
        if length:
            return None
        return (
            ArgumentCandidate(f"new {inner}[](0)", "zero", type_name),
            element,
        )
    uint = _UINT.fullmatch(type_name)
    if uint:
        width = int(uint.group(1) or "256")
        if width == 0 or width > 256 or width % 8:
            return None
        return (
            ArgumentCandidate("0", "zero", type_name),
            ArgumentCandidate("1", "one", type_name),
            ArgumentCandidate(f"type({type_name}).max", "max", type_name),
        )
    signed = _INT.fullmatch(type_name)
    if signed:
        width = int(signed.group(1) or "256")
        if width == 0 or width > 256 or width % 8:
            return None
        return (
            ArgumentCandidate("0", "zero", type_name),
            ArgumentCandidate("1", "one", type_name),
            ArgumentCandidate("-1", "boundary", type_name),
            ArgumentCandidate(f"type({type_name}).min", "min", type_name),
        )
    if type_name == "bool":
        return (
            ArgumentCandidate("false", "zero", type_name),
            ArgumentCandidate("true", "one", type_name),
        )
    if type_name == "address":
        return (
            ArgumentCandidate("address(0)", "zero", type_name),
            ArgumentCandidate("address(1)", "one", type_name),
        )
    if type_name == "bytes":
        return (
            ArgumentCandidate('hex""', "zero", type_name),
            ArgumentCandidate('hex"01"', "one", type_name),
        )
    fixed = _BYTES_N.fullmatch(type_name)
    if fixed:
        width = int(fixed.group(1))
        if not 1 <= width <= 32:
            return None
        return (ArgumentCandidate(f"{type_name}(0)", "zero", type_name),)
    if type_name.startswith("enum "):
        return (ArgumentCandidate("0", "zero", type_name),)
    if type_name.startswith("contract "):
        return (ArgumentCandidate("address(1)", "contract-address", type_name),)
    if type_name.startswith("(") and type_name.endswith(")"):
        inner = parameter_types("f" + type_name)
        if inner is None:
            return None
        pieces: list[str] = []
        for item in inner:
            found = _candidates(item)
            if not found:
                return None
            pieces.append(found[0].literal)
        return (ArgumentCandidate("(" + ", ".join(pieces) + ")", "tuple", type_name),)
    if type_name[:1].isupper():
        return (ArgumentCandidate("address(1)", "contract-address", type_name),)
    return None


def _split_args(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts
