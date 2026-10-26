"""The LLM tier of `cv advise`: cheap first, validated always, offline in tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from jobbot.cv.advisor import Advice, Axis, Target, TargetKind, advise
from jobbot.cv.llm_advice import (
    Budget,
    LlmRewriter,
    Tier,
    plan_prompt,
)
from jobbot.models.candidate import Candidate
from tests.fixtures.profile import nurse_profile_dict, sample_profile_dict


@dataclass
class FakeChatModel:
    """Stands in for a LangChain chat model: no network, no key, no cost."""

    replies: list[str]
    calls: list[str] = field(default_factory=list)
    model_name: str = "fake-model"

    def invoke(self, messages: object) -> object:
        self.calls.append(str(messages))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]

        class _Message:
            content = reply

        return _Message()


def _bullet_advice(candidate: Candidate) -> Advice:
    exp = candidate.experience[0]
    return Advice(
        axis=Axis.LANGUAGE,
        target=Target(
            kind=TargetKind.ACHIEVEMENT,
            experience_id=exp.id,
            achievement_id=exp.achievements[0].id,
        ),
        what="lead with the verb",
        before=exp.achievements[0].text,
        why="the reader gives each line a second",
    )


def _candidate() -> Candidate:
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = (
        "Fui responsable de coordinar el turno de urgencias con 8 personas."
    )
    return Candidate.model_validate(raw)


def _json(text: str) -> str:
    return '{"rewrite": "' + text + '"}'


def test_a_valid_rewrite_is_taken() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("Coordiné el turno de urgencias con 8 personas.")])
    rewriter = LlmRewriter(chat_model=model)

    improved = rewriter.improve(_bullet_advice(candidate), candidate)

    assert improved is not None
    assert improved.after == "Coordiné el turno de urgencias con 8 personas."
    assert len(model.calls) == 1


def test_an_invented_technology_is_discarded_and_tier_zero_stands() -> None:
    """The model may write better prose; it may not add a fact."""
    candidate = _candidate()
    model = FakeChatModel(
        replies=[_json("Coordiné el turno con 8 personas usando Kubernetes y Airflow.")]
    )
    rewriter = LlmRewriter(chat_model=model)

    improved = rewriter.improve(_bullet_advice(candidate), candidate)

    assert improved is None
    assert rewriter.rejections
    assert "kubernetes" in rewriter.rejections[0].casefold()


def test_a_rewrite_that_loses_a_figure_is_discarded() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("Coordiné el turno de urgencias con el equipo.")])
    rewriter = LlmRewriter(chat_model=model)

    assert rewriter.improve(_bullet_advice(candidate), candidate) is None
    assert "8" in rewriter.rejections[0]


def test_unparseable_output_is_discarded_without_raising() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=["I'm sorry, I cannot help with that."])
    rewriter = LlmRewriter(chat_model=model)

    assert rewriter.improve(_bullet_advice(candidate), candidate) is None


def test_the_prompt_carries_the_bullet_and_not_the_whole_cv() -> None:
    candidate = _candidate()
    advice = _bullet_advice(candidate)

    prompt = plan_prompt(advice, candidate)

    assert advice.before in prompt
    other = candidate.education[0].institution
    assert other not in prompt, "only the unit being reworded is sent"
    assert len(prompt) < 1200


def test_contact_data_never_enters_the_prompt() -> None:
    raw = nurse_profile_dict()
    raw["personal"]["email"] = "marta.real@empresa-real.cl"
    raw["personal"]["phone"] = "+56 9 8765 4321"
    raw["experience"][0]["achievements"][0]["text"] = (
        "Fui responsable del turno; contacto marta.real@empresa-real.cl o +56 9 8765 4321."
    )
    candidate = Candidate.model_validate(raw)

    prompt = plan_prompt(_bullet_advice(candidate), candidate)

    assert "marta.real@empresa-real.cl" not in prompt
    assert "+56 9 8765 4321" not in prompt


def test_the_cache_prevents_a_second_call(tmp_path: Path) -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("Coordiné el turno de urgencias con 8 personas.")])
    rewriter = LlmRewriter(chat_model=model, cache_dir=tmp_path)

    first = rewriter.improve(_bullet_advice(candidate), candidate)
    second = LlmRewriter(chat_model=model, cache_dir=tmp_path).improve(
        _bullet_advice(candidate), candidate
    )

    assert first is not None and second is not None
    assert first.after == second.after
    assert len(model.calls) == 1, "the same paragraph must not be sent twice"


def test_a_cache_from_another_model_is_not_reused(tmp_path: Path) -> None:
    candidate = _candidate()
    good = _json("Coordiné el turno de urgencias con 8 personas.")
    first_model = FakeChatModel(replies=[good], model_name="model-a")
    LlmRewriter(chat_model=first_model, cache_dir=tmp_path).improve(
        _bullet_advice(candidate), candidate
    )

    other = FakeChatModel(replies=[good], model_name="model-b")
    LlmRewriter(chat_model=other, cache_dir=tmp_path).improve(_bullet_advice(candidate), candidate)

    assert len(other.calls) == 1


def test_the_budget_stops_the_run() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("Coordiné el turno de urgencias con 8 personas.")])
    rewriter = LlmRewriter(chat_model=model, budget=Budget(max_calls=1))

    rewriter.improve(_bullet_advice(candidate), candidate)
    blocked = rewriter.improve(_bullet_advice(candidate), candidate)

    assert blocked is None
    assert len(model.calls) == 1
    assert rewriter.budget.exhausted


def test_a_prompt_over_the_character_cap_is_not_sent() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("nunca llega")])
    rewriter = LlmRewriter(chat_model=model, budget=Budget(max_calls=5, max_chars=10))

    assert rewriter.improve(_bullet_advice(candidate), candidate) is None
    assert model.calls == []


def test_dry_run_says_what_it_would_send_and_calls_nobody() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("no debería usarse")])
    rewriter = LlmRewriter(chat_model=model, dry_run=True)

    planned = rewriter.improve(_bullet_advice(candidate), candidate)

    assert planned is None
    assert model.calls == []
    assert rewriter.planned
    assert rewriter.planned[0].chars > 0
    assert rewriter.total_planned_chars == rewriter.planned[0].chars


def test_the_reasoning_tier_is_only_tried_after_the_cheap_one_fails() -> None:
    candidate = _candidate()
    model = FakeChatModel(
        replies=[
            _json("Coordiné el turno con 8 personas usando Kubernetes."),
            _json("Coordiné el turno de urgencias con 8 personas."),
        ]
    )
    rewriter = LlmRewriter(chat_model=model, deep=True)

    improved = rewriter.improve(_bullet_advice(candidate), candidate)

    assert improved is not None
    assert len(model.calls) == 2
    assert rewriter.tiers_used == [Tier.LLM, Tier.LLM_DEEP]


def test_without_the_reasoning_flag_one_call_is_all_it_gets() -> None:
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("Coordiné el turno usando Kubernetes.")])
    rewriter = LlmRewriter(chat_model=model, deep=False)

    assert rewriter.improve(_bullet_advice(candidate), candidate) is None
    assert len(model.calls) == 1


def test_a_note_is_never_sent_to_a_model() -> None:
    """Tier 1 exists to reword text, not to answer questions for you."""
    candidate = _candidate()
    model = FakeChatModel(replies=[_json("algo")])
    rewriter = LlmRewriter(chat_model=model)

    note = Advice(
        axis=Axis.LAYOUT,
        target=Target(kind=TargetKind.PROFILE),
        what="decide what goes on the first screen",
        why="from the form",
    )

    assert rewriter.improve(note, candidate) is None
    assert model.calls == []


def test_no_model_available_means_tier_zero_and_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suite runs with no API key and without the llm extra installed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    candidate = _candidate()

    rewriter = LlmRewriter()

    assert not rewriter.available
    assert rewriter.improve(_bullet_advice(candidate), candidate) is None
    assert advise(candidate, limit=2), "the deterministic advisor keeps working"


def test_the_deterministic_run_never_touches_a_model() -> None:
    """Tier 0 is the default, and it is the whole default: it cannot even call out."""
    import ast
    import inspect

    from jobbot.cv import advisor

    tree = ast.parse(inspect.getsource(advisor))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any("langchain" in name for name in imported)
    assert not any("llm" in name for name in imported)
    assert not any("openai" in name for name in imported)


def test_improving_a_suggestion_keeps_its_identity() -> None:
    """The log must recognise a suggestion whether or not a model reworded it."""
    candidate = Candidate.model_validate(sample_profile_dict())
    raw = nurse_profile_dict()
    raw["experience"][0]["achievements"][0]["text"] = "Fui responsable de coordinar el turno."
    candidate = Candidate.model_validate(raw)
    original = _bullet_advice(candidate)
    model = FakeChatModel(replies=[_json("Coordiné el turno.")])

    improved = LlmRewriter(chat_model=model).improve(original, candidate)

    assert improved is not None
    assert improved.target == original.target
    assert improved.axis is original.axis


def _cli_workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    raw = (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8")
    (tmp_path / "data" / "profile.yaml").write_text(
        raw.replace(
            "Modelos de customer analytics",
            "Fui responsable de modelos de customer analytics",
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_cv_advise_dry_run_prices_the_run_without_calling(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _cli_workspace(tmp_path, project_root, monkeypatch)
    import jobbot.cv.llm_advice as llm_advice
    from jobbot.cli import run_cli
    from jobbot.exit_codes import SUCCESS

    def _never(**_kwargs: object) -> object:
        raise AssertionError("--dry-run must not build a model")

    monkeypatch.setattr(llm_advice, "_build_chat_model", _never)

    code = run_cli(["cv", "advise", "--llm", "--dry-run"], standalone_mode=False)
    out = " ".join(capsys.readouterr().out.split())

    assert code == SUCCESS
    assert "nothing was sent" in out
    assert "characters" in out
    assert "Drop --dry-run to spend it" in out


def test_cv_advise_without_the_extra_falls_back_and_says_so(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _cli_workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.exit_codes import SUCCESS

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = run_cli(["cv", "advise", "--llm"], standalone_mode=False)
    out = capsys.readouterr().out

    assert code == SUCCESS
    assert "No LLM reachable" in out
