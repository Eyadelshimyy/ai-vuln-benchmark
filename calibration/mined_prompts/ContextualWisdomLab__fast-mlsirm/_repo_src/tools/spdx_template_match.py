"""Match a license text against an SPDX license-list-data matching template (stdlib only).

Used to review texts before their normalized sha256 is added to the verifier's reviewed map;
the verifier itself only trusts those exact hashes.

Guidelines applied: whitespace, case and quote/dash equivalence; code comment indicators
(/*, *, //, #) ignored; a license title at the start ignored; <<var>> matches its `match`
regex; <<beginOptional>>...<<endOptional>> may be absent; everything else must match word for word.

Strictness beyond the template (the SPDX `match` regexes are very permissive):
- the `copyright` var may hold only title/copyright lines (".{0,5000}" would otherwise absorb any
  preceding text, even a whole other license);
- any other unbounded var (".+", ".*") may capture at most UNBOUNDED_VAR_MAX characters.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EQUIV = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
HEADER_LINE = re.compile(
    r"^((the )?mit license( \(mit\))?|[a-z0-9 .\-/]{1,40} licen[cs]e( v?\d[\w.]*)?|-{3,}|={3,}"
    r"|copyright\b.*|(\(c\)|©)\s*\d{4}.*|all rights reserved\.?)$", re.I)
# SPDX Matching Guidelines: code comment indicators and separators are ignored.
COMMENT_MARKER = re.compile(r"^\s*(/\*+|\*+/|\*+(?=\s|$)|//+|#+(?=\s|$))\s?|\s*\*+/\s*$", re.M)
UNBOUNDED_VAR_MAX = 200


def compile_template(tpl: str):
    names, stack, n = [], [[]], 0
    for tok in re.split(r"(<<.*?>>)", tpl, flags=re.S):
        if tok == "<<beginOptional>>":
            stack.append([])
        elif tok == "<<endOptional>>":
            body = "".join(stack.pop())
            n += 1
            names.append((f"opt{n}", "optional", "", ""))
            stack[-1].append(rf"(?:\s*(?P<opt{n}>{body}))?")
        elif tok.startswith("<<var;"):
            attrs = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', tok))
            n += 1
            names.append((f"var{n}", "var:" + attrs["name"], attrs.get("original", ""), attrs["match"]))
            stack[-1].append(rf"\s*(?P<var{n}>{attrs['match']})")
        else:
            words = tok.translate(EQUIV).split()
            if words:
                stack[-1].append(r"\s*" + r"\s+".join(re.escape(w) for w in words))
    return re.compile(r"\s*" + "".join(stack[0]) + r"\s*", re.I | re.S), names


def match(text: str, tpl: str, strict: bool = True) -> list[dict] | None:
    """Return the var/optional spans that differ from the template's original, or None."""
    # ponytail: length guard instead of a regex engine with backtracking limits; a text longer than
    # the template plus the widest vars cannot match, and huge combined files otherwise backtrack for minutes.
    if len(text.split()) > 2 * len(re.sub(r"<<.*?>>", " ", tpl, flags=re.S).split()) + 1000:
        return None
    rx, names = compile_template(tpl)
    m = rx.fullmatch(COMMENT_MARKER.sub("", text).replace("``", '"').replace("''", '"').translate(EQUIV))
    if not m:
        return None
    if strict:
        for g, kind, _, pattern in names:
            value = m.group(g)
            if not value:
                continue
            if kind == "var:copyright":
                if not all(not line.strip() or HEADER_LINE.match(line.strip()) for line in value.splitlines()):
                    return None
            elif kind.startswith("var:") and re.search(r"\.[+*]|\.\{0,\d{3,}\}", pattern) \
                    and len(" ".join(value.split())) > UNBOUNDED_VAR_MAX:
                return None
    return [{"span": kind, "template_original": " ".join(orig.split()), "text": " ".join((m.group(g) or "").split())}
            for g, kind, orig, _ in names
            if m.group(g) is not None and " ".join(m.group(g).split()) != " ".join(orig.split())]


if __name__ == "__main__":
    spans = match(Path(sys.argv[1]).read_text(), Path(sys.argv[2]).read_text())
    print("NO MATCH" if spans is None else spans)
    sys.exit(0 if spans is not None else 1)
