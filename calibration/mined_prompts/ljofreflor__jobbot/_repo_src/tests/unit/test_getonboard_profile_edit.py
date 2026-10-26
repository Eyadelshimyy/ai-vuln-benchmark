"""Writing the permanent Get on Board profile: plan first, never delete facts."""

from __future__ import annotations

from typing import Any

from jobbot.adapters.getonboard.profile_edit import (
    FIELDS,
    SAVE_BUTTON,
    apply_writes,
    desired_from_fields,
    plan_writes,
    read_current,
)

OLD_TEXT = "Me he dedicado a trabajos numéricos en matlab y C++."
NEW_TEXT = "Senior Data Scientist con 12 años de experiencia.\n\nStack: Python, SQL, R."


class FakeTrix:
    """A Trix editor: text is read from the node, written through editor.loadHTML."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.loaded_html: str | None = None

    def inner_text(self) -> str:
        return self.text

    def input_value(self) -> str:  # pragma: no cover - trix reads via inner_text
        raise AssertionError("a trix editor must not be read as an input")

    def evaluate(self, expression: str, arg: Any = None) -> Any:
        assert "editor.loadHTML" in expression, expression
        self.loaded_html = str(arg)
        return None

    def fill(self, value: str) -> None:  # pragma: no cover - would silently no-op
        raise AssertionError("fill() does not reach the Trix hidden input")


class FakeInput:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def inner_text(self) -> str:  # pragma: no cover
        raise AssertionError("a plain input must be read with input_value()")

    def input_value(self) -> str:
        return self.value

    def evaluate(self, expression: str, arg: Any = None) -> Any:  # pragma: no cover
        raise AssertionError("a plain input needs no JS")

    def fill(self, value: str) -> None:
        self.value = value


class FakeEditPage:
    def __init__(self, elements: dict[str, Any]) -> None:
        self.elements = elements
        self.clicked: list[str] = []
        self.visited: list[str] = []

    def goto(self, url: str, **_: Any) -> None:
        self.visited.append(url)

    def query_selector(self, selector: str) -> Any:
        return self.elements.get(selector)

    def click(self, selector: str, **_: Any) -> None:
        self.clicked.append(selector)

    def wait_for_timeout(self, _timeout: float) -> None:
        return None


def _page(
    *, description: str = "", professional: str = OLD_TEXT, academic: str = ""
) -> FakeEditPage:
    by_key = {
        "description_es": FakeInput(description),
        "professional_es": FakeTrix(professional),
        "academic_background_es": FakeTrix(academic),
    }
    return FakeEditPage({field.selector: by_key[field.key] for field in FIELDS})


def test_plan_reports_what_the_portal_holds_today() -> None:
    page = _page()
    desired = desired_from_fields(
        "Senior Data Scientist", NEW_TEXT, "Magíster en Estadística — PUC."
    )

    writes = plan_writes(read_current(page), desired)

    described = " ".join(write.describe() for write in writes)
    assert "replace Perfil profesional y experiencia laboral" in described
    assert "fill Formación académica y estudios" in described
    assert "fill Descripción profesional" in described
    assert page.clicked == [], "planning must not touch the form"


def test_unchanged_fields_are_not_rewritten() -> None:
    page = _page(description="Senior Data Scientist", professional=NEW_TEXT, academic="PUC.")
    desired = desired_from_fields("Senior Data Scientist", NEW_TEXT, "PUC.")

    assert plan_writes(read_current(page), desired) == []


def test_whitespace_only_differences_are_not_a_change() -> None:
    page = _page(professional="Senior   Data Scientist\n\nStack: Python.")
    desired = desired_from_fields("", "Senior Data Scientist\n\nStack: Python.", "")

    assert plan_writes(read_current(page), desired) == []


def test_an_empty_draft_never_wipes_the_portal() -> None:
    page = _page(description="Algo escrito", professional=OLD_TEXT, academic="Estudios.")

    assert plan_writes(read_current(page), desired_from_fields("", "", "")) == []


def test_apply_writes_trix_as_html_and_saves_once() -> None:
    page = _page()
    desired = desired_from_fields("Senior Data Scientist", NEW_TEXT, "PUC.")

    done = apply_writes(page, plan_writes(read_current(page), desired))

    assert {write.field.key for write in done} == {
        "description_es",
        "professional_es",
        "academic_background_es",
    }
    professional = page.elements[
        next(f.selector for f in FIELDS if f.key == "professional_es")
    ]
    assert professional.loaded_html is not None
    # Paragraphs survive as HTML blocks; the portal stores rich text, not plain text.
    assert professional.loaded_html.count("<div>") == 2
    description = page.elements[next(f.selector for f in FIELDS if f.key == "description_es")]
    assert description.value == "Senior Data Scientist"
    assert page.clicked == [SAVE_BUTTON]


def test_nothing_to_write_means_nothing_is_saved() -> None:
    page = _page(description="X", professional=OLD_TEXT, academic="Y")

    assert apply_writes(page, []) == []
    assert page.clicked == []


def test_html_in_the_draft_is_escaped_not_injected() -> None:
    page = _page()
    desired = desired_from_fields("", "Trabajé con <script>alert(1)</script> y R.", "")

    apply_writes(page, plan_writes(read_current(page), desired))

    professional = page.elements[
        next(f.selector for f in FIELDS if f.key == "professional_es")
    ]
    assert professional.loaded_html is not None
    assert "<script>" not in professional.loaded_html
    assert "&lt;script&gt;" in professional.loaded_html
