from __future__ import annotations

from pathlib import Path

from fpga import _vrp
from fpga.liaison import _records_heads  # pyright: ignore[reportPrivateUsage]
from fpga.records import records_dir


def test_records_heads_are_genesis_then_skip_a_broken_chain(tmp_path: Path) -> None:
    heads, broken = _records_heads(tmp_path)
    assert broken == []
    assert heads == dict.fromkeys(_vrp.LOG_FILES, _vrp.GENESIS)
    log = records_dir(tmp_path) / _vrp.LOG_FILES["upstream_read"]
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('{"schema_version": 2, "event_id": "' + "1" * 64 + '"}\n', encoding="utf-8")
    heads, broken = _records_heads(tmp_path)
    assert broken == ["upstream_read"]
    assert "upstream_read" not in heads
