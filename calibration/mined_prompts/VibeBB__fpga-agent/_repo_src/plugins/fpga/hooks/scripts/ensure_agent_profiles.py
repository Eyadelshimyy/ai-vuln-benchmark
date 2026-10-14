#!/usr/bin/env python3
"""SessionStart hook: provision the vibebb-* agent profile for this plugin.

OpenHands Agent Canvas resolves a conversation's agent from
``~/.openhands/agent-profiles/<name>.json`` (schema_version 3,
``OpenHandsAgentProfile``). A scoped profile gives a design session the
least-privilege surface VibeBB expects: only this plugin's MCP server, no
secrets, and the ``vibebb-author`` lane as the writing model. This hook
writes ``vibebb-<plugin>.json`` when — and only when — it is absent, so
operators can edit routing, condenser, verification, or model routing
fields afterwards without the hook overwriting their choices.

Fields deliberately left at defaults/not set:

- ``tools``: ``None`` — the server's standard tool set (terminal, file
  editor, task, finish); VibeBB sub-agents ride the same conversation and
  need ``task`` available.
- ``persona``/``disabled_skills``: keep the stock persona and every
  discovered skill — sisters coordinate through each other's skills.
- ``verification`` (native critic): left off; VibeBB keeps its own review
  lane plus deterministic gates as the verdict authority, and a second
  always-on reviewer would double review cost for no verdict.
- ``meta_profile_ref``/``enable_classify_and_switch_llm_tool``: off; the
  explicit author/review lanes already declare routing.
- ``condenser``/``tool_concurrency_limit``: model defaults; operators may
  tune them for long design sessions.

Advisory only: reads nothing but the plugin path, writes absent files,
prints a JSON finding, always exits 0.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

_SCHEMA_VERSION = 3
_SUFFIX = (
    "This profile scopes to the VibeBB {name} plugin. Deterministic gates "
    "are the only verdict authority; generated artifacts under out/ are "
    "read-only — change design inputs and regenerate instead of editing "
    "them by hand."
)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _profile(plugin_name: str) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "id": str(uuid.uuid4()),
        "name": f"vibebb-{plugin_name}",
        "revision": 0,
        "agent_kind": "openhands",
        "llm_profile_ref": "vibebb-author",
        "mcp_server_refs": [plugin_name],
        "secret_refs": [],
        "system_message_suffix": _SUFFIX.format(name=plugin_name),
    }


def main() -> int:
    plugin_name = Path(__file__).resolve().parents[2].name
    store = Path.home() / ".openhands" / "agent-profiles"
    dest = store / f"vibebb-{plugin_name}.json"
    findings: list[str] = []
    provisioned = False

    if dest.is_file():
        findings.append(f"vibebb-{plugin_name} already present")
    else:
        try:
            _atomic_write(dest, json.dumps(_profile(plugin_name), indent=2) + "\n")
            provisioned = True
            findings.append(f"provisioned vibebb-{plugin_name}")
        except OSError as exc:
            findings.append(f"could not write {dest}: {exc}")

    print(
        json.dumps(
            {
                "hook": "ensure-agent-profiles",
                "profile": f"vibebb-{plugin_name}",
                "provisioned": provisioned,
                "findings": findings,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
