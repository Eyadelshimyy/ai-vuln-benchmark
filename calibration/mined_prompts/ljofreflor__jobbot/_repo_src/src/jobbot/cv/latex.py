"""LaTeX escaping helpers."""

from __future__ import annotations

import re

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_ESCAPE_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_LATEX_SPECIALS, key=len, reverse=True))
)


def escape_latex(text: str | None) -> str:
    """Escape characters that are special in LaTeX text mode.

    ``None`` becomes an empty string so optional profile fields (city,
    location, …) do not crash Jinja's ``latex`` filter mid-render.
    """
    if text is None:
        return ""
    return _ESCAPE_RE.sub(lambda m: _LATEX_SPECIALS[m.group(0)], text)


def escape_latex_url(url: str) -> str:
    r"""Escape a URL for the target of ``\href`` used inside another macro's argument."""
    return url.replace("%", r"\%").replace("#", r"\#")


def escape_latex_multiline(text: str | None) -> str:
    """Escape LaTeX and normalize whitespace for paragraph fields."""
    if text is None:
        return ""
    cleaned = " ".join(text.split())
    return escape_latex(cleaned)
