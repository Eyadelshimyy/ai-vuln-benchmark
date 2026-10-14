from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "fpga"
    / "hooks"
    / "scripts"
    / "ensure_llm_profiles.py"
)
UNVERIFIED_FINDING = (
    "vibebb-review vision capability unverified (SDK not importable in the hook environment)"
)


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ensure_llm_profiles_vision", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_home(tmp_path: Path, review: dict[str, object]) -> tuple[Path, Path]:
    home = tmp_path / "home"
    config = home / ".openhands"
    profiles = config / "profiles"
    profiles.mkdir(parents=True)
    (config / "settings.json").write_text(
        json.dumps({"active_profile": "active"}), encoding="utf-8"
    )
    (profiles / "active.json").write_text(json.dumps({"model": "source"}), encoding="utf-8")
    (profiles / "vibebb-author.json").write_text(
        json.dumps({"model": "author", "disable_vision": True}),
        encoding="utf-8",
    )
    review_path = profiles / "vibebb-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return home, review_path


def _fake_llm(active: bool, calls: list[dict[str, object]]) -> type[object]:
    class FakeLLM:
        def __init__(
            self,
            *,
            model: str,
            usage_id: str,
            capability_overrides: dict[str, bool | str],
        ) -> None:
            calls.append(
                {
                    "model": model,
                    "usage_id": usage_id,
                    "capability_overrides": capability_overrides,
                }
            )

        def vision_is_active(self) -> bool:
            return active

    return FakeLLM


def _run(
    hook: ModuleType,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[dict[str, object], str]:
    monkeypatch.setenv("HOME", str(home))
    assert hook.main() == 0
    output = capsys.readouterr().out
    return cast(dict[str, object], json.loads(output)), output


def _vision(payload: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], payload["vision"])


def _findings(payload: dict[str, object]) -> list[str]:
    return cast(list[str], payload["findings"])


def test_disabled_review_is_reported_without_leaking_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "profile-secret-disabled"
    home, _ = _profile_home(
        tmp_path, {"model": "review", "disable_vision": True, "api_key": secret}
    )
    calls: list[dict[str, object]] = []
    hook = _load_hook()
    monkeypatch.setattr(hook, "LLM", _fake_llm(True, calls))

    payload, output = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "disabled"
    assert (
        "vibebb-review is not vision-capable (disabled); rendered-image review "
        "is text-only — point ~/.openhands/profiles/vibebb-review.json at a "
        "vision-capable model"
    ) in _findings(payload)
    assert calls == []
    assert secret not in output


def test_existing_profiles_are_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, review_path = _profile_home(
        tmp_path, {"model": "review", "api_key": "profile-secret-review"}
    )
    author_path = home / ".openhands" / "profiles" / "vibebb-author.json"
    author_before = author_path.read_text(encoding="utf-8")
    review_before = review_path.read_text(encoding="utf-8")
    hook = _load_hook()
    monkeypatch.setattr(hook, "LLM", _fake_llm(True, []))

    payload, output = _run(hook, home, monkeypatch, capsys)

    assert payload["missing"] == []
    assert author_path.read_text(encoding="utf-8") == author_before
    assert review_path.read_text(encoding="utf-8") == review_before
    assert "profile-secret-review" not in output


def test_unsupported_review_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, _ = _profile_home(tmp_path, {"model": "text-only"})
    calls: list[dict[str, object]] = []
    hook = _load_hook()
    monkeypatch.setattr(hook, "LLM", _fake_llm(False, calls))

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "unsupported"
    assert (
        "vibebb-review is not vision-capable (unsupported); rendered-image "
        "review is text-only — point ~/.openhands/profiles/vibebb-review.json "
        "at a vision-capable model"
    ) in _findings(payload)
    assert len(calls) == 1


def test_litellm_proxy_probe_omits_credentials_and_does_not_make_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "profile-secret-proxy"
    model = "litellm_proxy/provider/model"
    home, _ = _profile_home(
        tmp_path,
        {
            "model": model,
            "base_url": "https://proxy.invalid",
            "api_key": secret,
            "capability_overrides": {"vision": True},
        },
    )
    calls: list[dict[str, object]] = []
    hook = _load_hook()
    monkeypatch.setattr(hook, "LLM", _fake_llm(True, calls))

    payload, output = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "active"
    assert calls == [
        {
            "model": model,
            "usage_id": "vision-probe",
            "capability_overrides": {"vision": True},
        }
    ]
    assert secret not in output
    assert "base_url" not in output
    assert "api_key" not in output


def test_sdk_probe_errors_are_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, _ = _profile_home(tmp_path, {"model": "review"})
    hook = _load_hook()

    class BrokenLLM:
        def __init__(
            self,
            *,
            model: str,
            usage_id: str,
            capability_overrides: dict[str, bool | str],
        ) -> None:
            raise RuntimeError("probe unavailable")

    monkeypatch.setattr(hook, "LLM", BrokenLLM)

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "unverified"
    assert UNVERIFIED_FINDING in _findings(payload)


def test_missing_sdk_reports_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, _ = _profile_home(tmp_path, {"model": "review"})
    hook = _load_hook()
    monkeypatch.setattr(hook, "LLM", None)

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "unverified"
    assert UNVERIFIED_FINDING in _findings(payload)


def test_invalid_review_profile_is_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, review_path = _profile_home(tmp_path, {"model": "review"})
    review_path.write_text("{invalid", encoding="utf-8")
    hook = _load_hook()

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "unverified"
    assert UNVERIFIED_FINDING in _findings(payload)


def test_unreadable_review_profile_is_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home, review_path = _profile_home(tmp_path, {"model": "review"})
    original_read_text = Path.read_text

    def unreadable_read_text(path: Path, encoding: str | None = None) -> str:
        if path == review_path:
            raise OSError("unreadable")
        return original_read_text(path, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", unreadable_read_text)
    hook = _load_hook()

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert _vision(payload)["vibebb-review"] == "unverified"
    assert UNVERIFIED_FINDING in _findings(payload)


def _seeding_home(tmp_path: Path, settings: dict[str, object]) -> Path:
    home = tmp_path / "home"
    profiles = home / ".openhands" / "profiles"
    profiles.mkdir(parents=True)
    (home / ".openhands" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return home


def test_null_active_profile_falls_back_to_default_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _seeding_home(tmp_path, {"active_profile": None})
    profiles = home / ".openhands" / "profiles"
    (profiles / "default.json").write_text(json.dumps({"model": "gui-lane"}), encoding="utf-8")
    hook = _load_hook()

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert payload["missing"] == []
    seeded = json.loads((profiles / "vibebb-author.json").read_text(encoding="utf-8"))
    assert seeded["model"] == "gui-lane"
    assert "no active_profile in ~/.openhands/settings.json" in _findings(payload)
    assert "provisioned vibebb-author from profiles/default.json" in _findings(payload)


def test_null_active_profile_falls_back_to_agent_settings_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _seeding_home(
        tmp_path,
        {
            "active_profile": None,
            "agent_settings": {"llm": {"model": "inline-lane", "api_key": "s"}},
        },
    )
    profiles = home / ".openhands" / "profiles"
    hook = _load_hook()

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert payload["missing"] == []
    seeded = json.loads((profiles / "oracle.json").read_text(encoding="utf-8"))
    assert seeded["model"] == "inline-lane"
    assert "provisioned oracle from agent_settings.llm" in _findings(payload)


def test_active_profile_still_wins_over_default_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _seeding_home(tmp_path, {"active_profile": "active"})
    profiles = home / ".openhands" / "profiles"
    (profiles / "active.json").write_text(json.dumps({"model": "explicit"}), encoding="utf-8")
    (profiles / "default.json").write_text(json.dumps({"model": "gui-lane"}), encoding="utf-8")
    hook = _load_hook()

    _run(hook, home, monkeypatch, capsys)

    seeded = json.loads((profiles / "vibebb-author.json").read_text(encoding="utf-8"))
    assert seeded["model"] == "explicit"


def test_no_llm_source_reports_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = _seeding_home(tmp_path, {"active_profile": None})
    hook = _load_hook()

    payload, _ = _run(hook, home, monkeypatch, capsys)

    assert payload["missing"] == ["vibebb-author", "vibebb-review", "oracle"]
    assert "no LLM profile source found to clone" in _findings(payload)
