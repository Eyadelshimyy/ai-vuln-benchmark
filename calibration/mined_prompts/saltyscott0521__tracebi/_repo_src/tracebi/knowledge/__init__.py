"""The analyst knowledge base — good-practice lessons, delivered to any agent.

The framework's answer to "give a bring-your-own agent the best chance of doing
the analysis *right*, not just producing *a* number." Post-training bakes
judgement into a model's weights; we can't do that to someone else's agent, so
we bake it into the ENVIRONMENT instead — a curated body of lessons the agent
can reference, paired with the guardrails that refuse the wrong number and cite
the lesson at the moment of the mistake.

Two consumption forms, one body of content (they never diverge — both read these
files):

* **Package** — shipped with tracebi. ``list_lessons()`` / ``get_lesson()`` here
  back the ``tracebi knowledge`` CLI and the ``knowledge`` block of
  ``tracebi context`` (so any agent, over any harness, sees the curriculum and
  can pull a lesson on demand).
* **Skill** — ``skills/tracebi-analyst/SKILL.md`` references this same set, so a
  skill-aware agent activates the senior-analyst discipline by name.

A lesson is one Markdown file with a small frontmatter block::

    ---
    slug: ratio-of-totals
    title: A rate is a ratio of totals, never a mean of ratios
    when: computing any rate, %, margin, yield, or per-unit metric
    ---
    <body: the pitfall, the correct TraceBi pattern, a real example>

The lessons are the reference the feedback loop cites — never a binder that
rots unread (the fate of ``docs/agents/sop-*.md``, which no agent surface ever
pointed at). A rot-guard test keeps them true to the real API.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import NamedTuple, Optional

_LESSON_DIR = os.path.join(os.path.dirname(__file__), "lessons")


class Lesson(NamedTuple):
    """One good-practice lesson: its frontmatter plus the Markdown body."""

    slug: str
    title: str
    when: str
    body: str


def _parse(path: str) -> Lesson:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Frontmatter is the first `---`-delimited block; keep the parser tiny (no
    # YAML dependency) — the fields are flat `key: value` lines.
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    slug = meta.get("slug") or os.path.splitext(os.path.basename(path))[0]
    return Lesson(slug=slug, title=meta.get("title", slug),
                  when=meta.get("when", ""), body=body.strip())


@lru_cache(maxsize=1)
def _all() -> "dict[str, Lesson]":
    out: dict[str, Lesson] = {}
    if os.path.isdir(_LESSON_DIR):
        for name in sorted(os.listdir(_LESSON_DIR)):
            if name.endswith(".md"):
                lesson = _parse(os.path.join(_LESSON_DIR, name))
                out[lesson.slug] = lesson
    return out


def list_lessons() -> "list[Lesson]":
    """Every lesson, sorted by slug — bodies included but usually read for the
    frontmatter (slug/title/when) when building a cheap index."""
    return list(_all().values())


def get_lesson(slug: str) -> Optional[Lesson]:
    """One lesson by slug, or ``None`` if there is no such lesson."""
    return _all().get(slug)


def index() -> "list[dict[str, str]]":
    """The curriculum as ``[{slug, title, when}]`` — the cheap listing that
    rides in ``tracebi context`` so an agent knows what exists and when to
    reach for it, without paying for every body up front."""
    return [{"slug": ls.slug, "title": ls.title, "when": ls.when}
            for ls in list_lessons()]
