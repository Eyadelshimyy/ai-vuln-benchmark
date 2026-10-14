#!/usr/bin/env python3
"""SessionStart hook: provision the vibebb-* LLM profiles agents need.

Every wire sub-agent declares `model: vibebb-author` or `model:
vibebb-review`, resolved through the SDK's LLMProfileStore
(~/.openhands/profiles/<name>.json). A missing profile raises ValueError
at task spawn and silently disables task delegation. This hook clones the
conversation's `active_profile` into the three vibebb profile slots when
they are absent so `task` works out of the box; operators can then edit
the files to route each lane at a different model.

When `settings.json` has no `active_profile` (GUI-configured hosts write
the lane into `profiles/default.json` and `agent_settings.llm` instead),
the hook falls back to `profiles/default.json`, then to the inline
`agent_settings.llm` dict, so seeding still works there.

Advisory only: reads settings, writes absent files, prints a JSON
finding, always exits 0.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

_PROFILES = ("vibebb-author", "vibebb-review", "oracle")

if TYPE_CHECKING:
    from openhands.sdk.llm import LLM as _LLMType

os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
_sdk_llm: type[_LLMType] | None
try:
    from openhands.sdk.llm import LLM as _sdk_llm
except ImportError:
    _sdk_llm = None

LLM: type[_LLMType] | None = _sdk_llm


def _settings() -> dict[str, object]:
    path = Path.home() / ".openhands" / "settings.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return cast("dict[str, object]", data)


def _read_profile(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return cast("dict[str, object]", data)


def _vision_status(profile: dict[str, object]) -> str:
    """'disabled' | 'active' | 'unsupported' | 'unverified'."""
    if profile.get("disable_vision") is True:
        return "disabled"
    model = profile.get("model")
    if LLM is None or not isinstance(model, str) or not model:
        return "unverified"
    try:
        overrides = profile.get("capability_overrides")
        if isinstance(overrides, dict):
            capability_overrides = cast("dict[str, bool | str]", overrides)
        else:
            capability_overrides = {}
        llm = LLM(
            model=model,
            usage_id="vision-probe",
            capability_overrides=capability_overrides,
        )
        return "active" if llm.vision_is_active() else "unsupported"
    except Exception:
        return "unverified"


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


def _template(
    settings: dict[str, object], store: Path, findings: list[str]
) -> tuple[dict[str, object] | None, str | None]:
    """LLM config dict to clone into the vibebb lanes, and its source label.

    Order: settings.active_profile -> profiles/default.json (the lane the
    AgentCanvas GUI writes when active_profile stays null) ->
    settings.agent_settings.llm (inline config dict).
    """
    active = settings.get("active_profile")
    if isinstance(active, str) and active:
        src = store / f"{active}.json"
        candidate = _read_profile(src)
        if isinstance(candidate, dict):
            return candidate, active
        findings.append(f"active_profile {active!r} unreadable at {src}")
    else:
        findings.append("no active_profile in ~/.openhands/settings.json")
    candidate = _read_profile(store / "default.json")
    if isinstance(candidate, dict):
        return candidate, "profiles/default.json"
    agent_settings = settings.get("agent_settings")
    if isinstance(agent_settings, dict):
        llm = cast(
            "dict[str, object] | None",
            cast("dict[str, object]", agent_settings).get("llm"),
        )
        if llm is not None and llm.get("model"):
            return dict(llm), "agent_settings.llm"
    return None, None


def main() -> int:
    settings = _settings()
    findings: list[str] = []
    store = Path.home() / ".openhands" / "profiles"

    template, source = _template(settings, store, findings)
    if template is None:
        findings.append("no LLM profile source found to clone")
    else:
        for name in _PROFILES:
            dest = store / f"{name}.json"
            if dest.is_file():
                continue
            try:
                _atomic_write(dest, json.dumps(template, indent=2) + "\n")
                findings.append(f"provisioned {name} from {source}")
            except OSError as exc:
                findings.append(f"could not write {dest}: {exc}")

    missing = [n for n in _PROFILES if not (store / f"{n}.json").is_file()]
    # Vision capability only gates the VibeBB review lane; `oracle` is a
    # plain text consult profile, so probing it would just waste a call.
    vision = {
        name: _vision_status(profile) if profile is not None else "unverified"
        for name in ("vibebb-author", "vibebb-review")
        if (store / f"{name}.json").is_file()
        for profile in (_read_profile(store / f"{name}.json"),)
    }
    review_status = vision.get("vibebb-review")
    if review_status in {"disabled", "unsupported"}:
        findings.append(
            "vibebb-review is not vision-capable "
            f"({review_status}); rendered-image review is text-only — point "
            "~/.openhands/profiles/vibebb-review.json at a vision-capable model"
        )
    elif review_status == "unverified":
        findings.append(
            "vibebb-review vision capability unverified "
            "(SDK not importable in the hook environment)"
        )
    print(
        json.dumps(
            {
                "hook": "ensure-llm-profiles",
                "profiles": _PROFILES,
                "missing": missing,
                "vision": vision,
                "findings": findings,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
