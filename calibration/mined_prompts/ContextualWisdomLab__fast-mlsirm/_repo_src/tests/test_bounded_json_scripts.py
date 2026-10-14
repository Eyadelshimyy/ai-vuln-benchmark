"""Security, invariance, and delegation tests for bounded JSON readers."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
_bounded_json = importlib.import_module("scripts._bounded_json")
read_json_object = _bounded_json.read_json_object
parse_json_bounded = _bounded_json.parse_json_bounded

_SCRIPT_MODULES = (
    "scripts.build_benchmark_report",
    "scripts.build_buyer_packet",
    "scripts.build_commercial_release",
    "scripts.build_figma_evidence_sync",
    "scripts.build_pr_queue_governance",
    "scripts.build_procurement_due_diligence",
    "scripts.build_release_evidence_index",
    "scripts.sales_readiness",
)


class _HostileInt(int):
    """Integer subclass whose comparison callback must never run in validation."""

    comparisons = 0

    def __le__(self, other: object) -> bool:
        """Fail if package validation dispatches caller-controlled comparison."""
        type(self).comparisons += 1
        raise AssertionError("caller-controlled integer comparison executed")


class _HostileText(str):
    """String subclass whose encoding callback must never run before admission."""

    encodes = 0

    def encode(self, *args: object, **kwargs: object) -> bytes:
        """Fail if bounded parsing dispatches caller-controlled encoding."""
        type(self).encodes += 1
        raise AssertionError("caller-controlled text encoding executed")


def _write(path: Path, content: bytes) -> Path:
    """Write exact bytes and return ``path``."""
    path.write_bytes(content)
    return path


def test_exact_size_depth_strings_escapes_and_unicode(tmp_path: Path) -> None:
    """Exact limits accept valid objects without counting string delimiters."""
    content = json.dumps(
        {
            "text": "[{}] \\ slash",
            "escaped_quote": 'quote: " and delimiters: {[]}',
            "nested": {"items": ["한글"]},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    path = _write(tmp_path / "valid.json", content)
    assert read_json_object(
        path,
        max_bytes=len(content),
        max_depth=3,
    ) == json.loads(content)


def test_limit_plus_one_is_rejected(tmp_path: Path) -> None:
    """One byte above the configured inclusive byte limit fails closed."""
    path = _write(tmp_path / "oversized.json", b'{"a":1} ')
    with pytest.raises(ValueError, match="maximum allowed size 7 bytes"):
        read_json_object(path, max_bytes=7)


def test_depth_boundary_and_excess(tmp_path: Path) -> None:
    """The depth boundary is inclusive and the next level is rejected."""
    boundary = _write(tmp_path / "boundary.json", b'{"a":{"b":0}}')
    excessive = _write(tmp_path / "excessive.json", b'{"a":{"b":[]}}')
    assert read_json_object(boundary, max_depth=2) == {"a": {"b": 0}}
    with pytest.raises(ValueError, match="maximum allowed depth 2"):
        read_json_object(excessive, max_depth=2)


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("max_bytes", 0),
        ("max_bytes", True),
        ("max_bytes", "7"),
        ("max_depth", -1),
        ("max_depth", True),
        ("max_depth", 1.5),
    ),
)
def test_limits_require_positive_non_boolean_integers(
    tmp_path: Path,
    keyword: str,
    value: object,
) -> None:
    """Invalid limit configuration fails before JSON decoding."""
    path = _write(tmp_path / "input.json", b"{}")
    with pytest.raises(ValueError, match=f"{keyword} must be a positive integer"):
        read_json_object(path, **{keyword: value})


@pytest.mark.parametrize("keyword", ("max_bytes", "max_depth"))
def test_limit_subclasses_fail_before_comparison_callbacks(
    tmp_path: Path,
    keyword: str,
) -> None:
    """Caller-defined integer limits are rejected without comparison dispatch."""
    path = _write(tmp_path / "input.json", b"{}")
    _HostileInt.comparisons = 0

    with pytest.raises(ValueError, match=f"{keyword} must be a positive integer"):
        read_json_object(path, **{keyword: _HostileInt(8)})

    assert _HostileInt.comparisons == 0


def test_invalid_utf8_malformed_json_and_decoder_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encoding, syntax, and recursion failures remain deterministic."""
    invalid_utf8 = _write(
        tmp_path / "invalid-utf8.json", b'{"value":"' + bytes([0xFF]) + b'"}'
    )
    malformed = _write(tmp_path / "malformed.json", b'}{"value":1}')
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_json_object(invalid_utf8)
    with pytest.raises(ValueError, match="not valid JSON"):
        read_json_object(malformed)

    valid = _write(tmp_path / "valid-object.json", b"{}")

    def raise_recursion(*args: object, **kwargs: object) -> object:
        """Simulate a JSON decoder stack exhaustion for any decoder options."""
        raise RecursionError("decoder stack exhausted")

    monkeypatch.setattr(_bounded_json.json, "loads", raise_recursion)
    with pytest.raises(ValueError, match="decoder recursion capacity"):
        read_json_object(valid)


@pytest.mark.parametrize("module_name", _SCRIPT_MODULES)
def test_every_wrapper_rejects_non_object_json(
    module_name: str,
    tmp_path: Path,
) -> None:
    """Each script preserves its object-root contract through delegation."""
    module = importlib.import_module(module_name)
    path = _write(tmp_path / f"{module_name.rsplit('.', 1)[-1]}.json", b"[]")
    with pytest.raises(RuntimeError, match="must be an object"):
        module._read_json(path)


@pytest.mark.parametrize("module_name", _SCRIPT_MODULES)
def test_every_wrapper_delegates_to_shared_reader(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every governed reader delegates without duplicating parsing logic."""
    module = importlib.import_module(module_name)
    requested = tmp_path / "artifact.json"
    marker = {"bounded": True}
    observed: list[Path] = []

    def fake_reader(path: Path) -> dict[str, bool]:
        """Record the delegated path and return the sentinel object."""
        observed.append(path)
        return marker

    monkeypatch.setattr(module, "read_json_object", fake_reader)
    assert module._read_json(requested) is marker
    assert observed == [requested]


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW unavailable")
def test_symbolic_link_is_rejected(tmp_path: Path) -> None:
    """A symbolic-link leaf is never followed on supporting platforms."""
    target = _write(tmp_path / "target.json", b"{}")
    link = tmp_path / "input.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(link)


def test_directory_is_rejected(tmp_path: Path) -> None:
    """A directory descriptor is rejected as nonregular input."""
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    """A FIFO is opened nonblocking and rejected before any content read."""
    fifo = tmp_path / "input.pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(fifo)


def test_descriptor_path_replacement_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the path after descriptor open is detected deterministically."""
    path = _write(tmp_path / "input.json", b"{}")
    replacement = _write(tmp_path / "replacement.json", b'{"replacement":true}')
    real_read = _bounded_json.os.read
    replaced = False

    def replacing_read(file_descriptor: int, byte_count: int) -> bytes:
        """Replace the path after the first descriptor read."""
        nonlocal replaced
        chunk = real_read(file_descriptor, byte_count)
        if not replaced:
            replaced = True
            path.unlink()
            replacement.replace(path)
        return chunk

    monkeypatch.setattr(_bounded_json.os, "read", replacing_read)
    with pytest.raises(ValueError, match="path changed"):
        read_json_object(path)


def test_missing_path_and_identity_lookup_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open and post-open identity failures use content-free errors."""
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(tmp_path / "missing.json")

    path = _write(tmp_path / "input.json", b"{}")
    real_lstat = _bounded_json.os.lstat
    calls = 0

    def failing_lstat(value: Path) -> os.stat_result:
        """Simulate disappearance during the post-open identity check."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileNotFoundError
        return real_lstat(value)

    monkeypatch.setattr(_bounded_json.os, "lstat", failing_lstat)
    with pytest.raises(ValueError, match="path changed"):
        read_json_object(path)


def test_initial_identity_mismatch_and_nonregular_identity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity mismatch and a nonregular lstat result are rejected."""
    path = _write(tmp_path / "input.json", b"{}")
    other = _write(tmp_path / "other.json", b"{}")
    other_status = os.lstat(other)
    monkeypatch.setattr(_bounded_json.os, "lstat", lambda _: other_status)
    with pytest.raises(ValueError, match="path changed"):
        read_json_object(path)

    class NonRegularStatus:
        """Synthetic stat result that is not a regular file."""

        st_mode = 0
        st_dev = other_status.st_dev
        st_ino = other_status.st_ino

    monkeypatch.setattr(_bounded_json.os, "lstat", lambda _: NonRegularStatus())
    with pytest.raises(ValueError, match="stable regular file"):
        read_json_object(path)


def test_parse_json_bounded_valid() -> None:
    """It successfully parses a valid, bounded string."""
    assert parse_json_bounded('{"a": 1}') == {"a": 1}
    assert parse_json_bounded("[1, 2, 3]") == [1, 2, 3]


def test_parse_json_bounded_oversized() -> None:
    """It rejects strings that exceed the byte limit."""
    content = '{"a": 1}'
    with pytest.raises(ValueError, match="exceeds maximum allowed size 4 bytes"):
        parse_json_bounded(content, max_bytes=4)


def test_parse_json_bounded_deep() -> None:
    """It rejects strings that exceed the maximum depth."""
    content = '{"a": {"b": {"c": 1}}}'
    with pytest.raises(ValueError, match="exceeds maximum allowed depth 2"):
        parse_json_bounded(content, max_depth=2)


def test_parse_json_bounded_invalid_json() -> None:
    """It raises ValueError for invalid JSON syntax."""
    with pytest.raises(ValueError, match="is not valid JSON"):
        parse_json_bounded("{bad")


def test_parse_json_bounded_rejects_text_subclass_before_encoding() -> None:
    """Caller-defined text fails before its encoding callback can execute."""
    _HostileText.encodes = 0

    with pytest.raises(TypeError, match="content must be str"):
        parse_json_bounded(_HostileText("{}"))

    assert _HostileText.encodes == 0
