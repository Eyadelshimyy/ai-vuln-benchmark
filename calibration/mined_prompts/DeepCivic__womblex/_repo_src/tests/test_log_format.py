"""Structured JSON logging: formatter, context propagation, CLI selection."""
from __future__ import annotations

import json
import logging

import pytest

from womblex.cli._shared import setup_logging
from womblex.utils.log_format import JsonFormatter, log_context


def _record(msg: str = "hello %s", args: tuple = ("x",), **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("womblex.t", logging.WARNING, __file__, 1, msg, args, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_record_is_one_json_object_with_core_fields() -> None:
    out = json.loads(JsonFormatter().format(_record()))
    assert out["message"] == "hello x"
    assert out["level"] == "WARNING"
    assert out["logger"] == "womblex.t"
    assert "ts" in out


def test_context_block_and_extra_are_carried() -> None:
    fmt = JsonFormatter()
    with log_context(run_id="r1", job_id=7, stage="chunk"):
        out = json.loads(fmt.format(_record(source_hash="abc")))
    assert (out["run_id"], out["job_id"], out["stage"], out["source_hash"]) == ("r1", 7, "chunk", "abc")
    after = json.loads(fmt.format(_record()))
    assert "run_id" not in after


def test_none_context_values_are_omitted_and_unknown_keys_refused() -> None:
    with log_context(run_id="r1", stage=None):
        out = json.loads(JsonFormatter().format(_record()))
    assert "stage" not in out
    with pytest.raises(ValueError), log_context(bogus=1):
        pass


def test_exception_is_serialised() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    assert "RuntimeError: boom" in json.loads(JsonFormatter().format(rec))["exc_info"]


def test_setup_logging_selects_json_from_flag_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        setup_logging(log_format="json")
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        monkeypatch.setenv("WOMBLEX_LOG_FORMAT", "json")
        setup_logging()
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        with pytest.raises(ValueError):
            setup_logging(log_format="xml")
    finally:
        root.handlers[:] = saved
