"""Load plugins/fpga through the OpenHands SDK plugin loader and assert the
expected agents, skills, commands, hooks, and manifest version.

Exits 0 on success and prints a one-line summary; exits 1 listing every
mismatch. Intended for the `plugin-load` CI job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugins" / "fpga"

EXPECTED_AGENTS = {"fpga-architect", "fpga-blind-review", "fpga-developer", "fpga-review"}
EXPECTED_SKILLS = {
    "fpga-contract",
    "fpga-contract-rules",
    "fpga-device-pins",
    "fpga-external-libraries",
    "fpga-out-rules",
    "fpga-sibling-cooperation",
    "fpga-verification",
    "fpga-workflow",
}
EXPECTED_COMMANDS = {
    "build",
    "design",
    "doctor",
    "gates",
    "liaison",
    "pinmap",
    "records",
    "render",
    "simulate",
    "verify",
}
EXPECTED_SESSION_START_HOOKS = {
    "fpga-doctor",
    "intake-attachments",
    "ensure-llm-profiles",
    "ensure-agent-profiles",
    "require-records",
}
EXPECTED_USER_PROMPT_SUBMIT_HOOKS = {"intake-attachments"}
EXPECTED_PRE_TOOL_USE_HOOKS = {"deny-programming", "protect-generated", "safety-rail"}
EXPECTED_STOP_HOOKS = {"require-records", "report-fpga-status", "intake-attachments"}
EXPECTED_POST_TOOL_USE_HOOKS = {
    "record-image-observation",
    "record-vision-tool-event",
}


def _registered_tools() -> set[str]:
    """Import the builtin tool modules so their registrations exist."""
    import openhands.tools.preset.default  # pyright: ignore[reportMissingImports,reportMissingModuleSource]
    from openhands.sdk.tool.registry import (  # pyright: ignore[reportMissingImports,reportMissingModuleSource]
        list_registered_tools,
    )

    openhands.tools.preset.default.register_default_tools(enable_browser=False)
    import openhands.tools.glob.definition  # pyright: ignore[reportMissingImports,reportMissingModuleSource,reportUnusedImport]
    import openhands.tools.grep.definition  # pyright: ignore[reportMissingImports,reportMissingModuleSource,reportUnusedImport]
    import openhands.tools.task.definition  # pyright: ignore[reportMissingImports,reportMissingModuleSource,reportUnusedImport]
    from openhands.sdk.tool.builtins import (  # pyright: ignore[reportMissingImports,reportMissingModuleSource]
        BUILT_IN_TOOL_CLASSES,
    )

    # resolve_tool falls back to BUILT_IN_TOOL_CLASSES for unregistered names.
    return set(list_registered_tools()) | set(BUILT_IN_TOOL_CLASSES)


def check_plugin(plugin_dir: Path) -> list[str]:
    """Return a list of mismatch reasons (empty means OK)."""
    from openhands.sdk.plugin import (  # pyright: ignore[reportMissingImports,reportMissingModuleSource]
        Plugin,
    )

    reasons: list[str] = []
    try:
        plugin = Plugin.load(plugin_dir)
    except Exception as e:
        return [f"Plugin.load failed: {e}"]

    manifest = json.loads((plugin_dir / ".plugin" / "plugin.json").read_text(encoding="utf-8"))
    if plugin.manifest.version != manifest.get("version"):
        reasons.append(
            f"manifest version {plugin.manifest.version!r} != "
            f"plugin.json {manifest.get('version')!r}"
        )

    agents = {a.name for a in plugin.agents}
    if agents != EXPECTED_AGENTS:
        reasons.append(f"agents {sorted(agents)} != {sorted(EXPECTED_AGENTS)}")

    skills = {s.name for s in plugin.skills}
    if skills != EXPECTED_SKILLS:
        reasons.append(f"skills {sorted(skills)} != {sorted(EXPECTED_SKILLS)}")

    commands = {c.name for c in plugin.commands}
    if commands != EXPECTED_COMMANDS:
        reasons.append(f"commands {sorted(commands)} != {sorted(EXPECTED_COMMANDS)}")

    if plugin.hooks is not None:
        collected: dict[str, set[str]] = {
            "session_start": set(),
            "user_prompt_submit": set(),
            "pre_tool_use": set(),
            "stop": set(),
            "post_tool_use": set(),
        }
        for event_name in collected:
            groups: list[Any] = getattr(plugin.hooks, event_name, None) or []
            for group in groups:
                hooks: list[Any] = list(group.hooks)
                names = [h.name for h in hooks if h.name is not None]
                collected[event_name].update(names)
        expected_hooks = {
            "session_start": EXPECTED_SESSION_START_HOOKS,
            "user_prompt_submit": EXPECTED_USER_PROMPT_SUBMIT_HOOKS,
            "pre_tool_use": EXPECTED_PRE_TOOL_USE_HOOKS,
            "stop": EXPECTED_STOP_HOOKS,
            "post_tool_use": EXPECTED_POST_TOOL_USE_HOOKS,
        }
        for event_name, expected in expected_hooks.items():
            if collected[event_name] != expected:
                reasons.append(
                    f"{event_name} hooks {sorted(collected[event_name])} != {sorted(expected)}"
                )

    registered = _registered_tools()
    for agent in plugin.agents:
        for tool in agent.tools:
            if tool not in registered:
                reasons.append(f"agent {agent.name!r} tool {tool!r} not registered")
    for command in plugin.commands:
        for tool in command.allowed_tools:
            if tool not in registered:
                reasons.append(f"command {command.name!r} allowed-tool {tool!r} not registered")
    return reasons


def main() -> int:
    reasons = check_plugin(PLUGIN_DIR)
    if reasons:
        for reason in reasons:
            print(reason)
        return 1
    all_hooks = (
        EXPECTED_SESSION_START_HOOKS
        | EXPECTED_USER_PROMPT_SUBMIT_HOOKS
        | EXPECTED_PRE_TOOL_USE_HOOKS
        | EXPECTED_STOP_HOOKS
        | EXPECTED_POST_TOOL_USE_HOOKS
    )
    print(
        f"plugin-load OK: agents={{{','.join(sorted(EXPECTED_AGENTS))}}} "
        f"skills={{{','.join(sorted(EXPECTED_SKILLS))}}} "
        f"commands={{{','.join(sorted(EXPECTED_COMMANDS))}}} "
        f"hooks={{{','.join(sorted(all_hooks))}}}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
